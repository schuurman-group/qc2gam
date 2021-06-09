"""
A set of routines to parse Molden format orbitals
"""
import sys
import numpy as np
import moinfo
from molden import read_geom, read_basis

# MOLCAS orbital ordering. SEWARD only uses spherical orbitals,
# we'll assume for now sph2cart is ALWAYS called. So, just use
# GAMESS ordering
# s px py pz dxx dxy dxz dyy dyz dzz
# fxxx fxxy fxxz fxyy fxyz fxzz fyyy fyyz fyzz fzzz
ao_ordr     = [['s'],
               ['px','py','pz'],
               ['dxx','dxy','dxz','dyy','dyz','dzz'],
               ['fxxx','fxxy','fxxz','fxyy','fxyz',
                'fxzz','fyyy','fyyz','fyzz','fzzz']]

ao_ordr_sph = [['s'],
               ['px', 'py', 'pz'],
               ['d-2','d-1','d0','d1','d2'],
               ['f-3','f-2','f-1','f0','f1','f2','f3']]

ao_norm     = [[1.],
               [1.,1.,1.],
               [np.sqrt(3.),np.sqrt(3.),np.sqrt(3.), 
                np.sqrt(3.),np.sqrt(3.),np.sqrt(3.)],
               [np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
                np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
                np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.)]]

# how to convert from spherical to cartesian basis functions
# s -> s | px -> px | py -> py | pz -> pz
# d ordering: d2-, d1-, d0, d1+, d2+
# dxx -> -d0 + d2+
# dxy ->  d2-
# dxz ->  d1+
# dyy -> -d0 - d2+
# dyz ->  d1-
# dzz ->  d0
# f ordering: f3-, f2-, f1-, f0, f1+, f2+, f3+
# fxxx -> -f1+ + f3+
# fxxy -> -f1- + f3-
# fxxz -> -f0  + f2+
# fxyy -> -f1+ - f3+
# fxyz ->  f2-
# fxzz ->  f1+
# fyyy -> -f1- + f3-
# fyyz -> -f0  - f2+
# fyzz ->  f1-
# fzzz ->  f0
a        = 1./(2.*np.sqrt(3.))
b        = 1./2.
c        = 1./np.sqrt(15.)
d        = np.sqrt(3./5.)/2.
sph2cart = [
    # conversion for s orbitals
    [[[0], [1.]]],
    # conversion for p orbitals
    [[[0], [1.]], [[1], [1.]], [[2], [1.]]],  
    # conversion for d orbitals
    [[[2, 4], [-a,  b]], 
     [[0], [1.]],
     [[3], [1.]],
     [[2, 4], [-a, -b]],
     [[1], [1.]],
     [[2], [1./np.sqrt(3.)]]],
    # conversion for f orbitals
    [[[4, 6], [c, 0.]], 
     [[0, 2], [b, -d]],
     [[3, 5], [-d, b]],
     [[4, 6], [-d, b]],
     [[1],[1.]],
     [[4, 6], [-d, -b]],
     [[0, 2], [0, c]],
     [[3, 5], [-d, -b]],
     [[0, 2], [-b, -d]],
     [[3, 5], [c, 0.]]] 
   ]

def parse(geom_file, geom_ordr, basis_file, mo_file):
    """Parses a set of molcas input files.
    Molcas orbital files do not contain geometry or basis. Instead,
    a molden file is parsed. Only one of basis_file and geom_file
    need to be specified.
    """
    # set unset values
    if geom_file is None:
        geom_file = basis_file
    elif basis_file is None:
        basis_file = geom_file

    # read xyz coord file
    gam_geom = read_geom(geom_file)
    if geom_ordr is not None:
        gam_ordr = read_geom(geom_ordr)
        gam_geom.reorder(gam_ordr)

    # parse basis file to get basis set information
    in_cart, gam_basis = read_basis(basis_file, gam_geom)

    # parse mos file to pull out orbitals
    gam_mos = read_mos(mo_file, in_cart, gam_basis)

    return gam_basis, gam_mos


def read_mos(mocoef_file, in_cart, basis):
    """Reads a Molcas orbital file into moinfo into moinfo.Orbitals.
    So: in the future, we should also allow for a mapping array that
    allows for taking linear combinations of elements in order to convert
    from a spherically adapted basis to cartesians. That will come later
    though... first the easy stuff.
    """
    # slurp up the molcas file
    with open(mocoef_file, 'r') as mos:
        mo_file = mos.readlines()

    nline = len(mo_file)
    for il in range(nline):
        if '#INFO' in mo_file[il]:
            # lines after: 1. comment, 2. uhf, nsym, 0, 3. nbas, 4. nbas
            n_mos = int(mo_file[il+3])
            break

    has_orb = False
    for il in range(il+1, nline):
        if '#ORB' in mo_file[il]:
            has_orb = True
            break

    if not has_orb:
        raise IOError('MOs not found in input file.')

    imo = -1
    n_ao_all = []
    raw_orbs = []
    for il in range(il+1, nline):
        if 'ORBITAL' in mo_file[il]:
            imo += 1
            raw_orbs.append([])
            n_ao_all.extend([0])
        elif imo < 0:
            raise ValueError('#ORB section must start with ORBITAL')
        elif '#' in mo_file[il]:
            # new section, break
            break
        else:
            mo_row = mo_file[il].split()
            n_ao_all[-1] += len(mo_row)
            raw_orbs[-1] += [float(o) for o in mo_row]

    # make sure all mos are the same length
    uniq = list(set(n_ao_all))
    if len(uniq) != 1:
        raise ValueError('MOs have different numbers of AOs. Exiting...')
    else:
        n_aos = uniq[0]

    # create a numpy array to hold populations, if present
    imo = 0
    molc_occ = np.zeros(n_mos, dtype=float)
    has_occ = False
    for il in range(il, nline):
        if has_occ:
            if 'OCCUPATION' in mo_file[il]:
                continue
            elif '#' in mo_file[il]:
                # new section, break
                break
            else:
                occ_row = mo_file[il].split()
                molc_occ[imo:imo+len(occ_row)] = [float(o) for o in occ_row]
                imo += len(occ_row)
        else:
            if '#OCC' in mo_file[il]:
                has_occ = True

    if np.allclose(molc_occ, 0):
        molc_occ = None
    else:
        molc_occ = np.array(molc_occ)

    # determine if we're running in cartesian or spherically adapted
    # atomic basis functions
    if n_aos == basis.n_bf_cart:
        in_cart = True
    elif n_aos == basis.n_bf_sph:
        in_cart = False
    else:
        raise ValueError('Number of basis functions is not equal to: ' +
                         '{:d}'.format(uniq[0]))

    # MOLCAS requires an intermediate sort
    molc_orb = sort_molcas_orbs(in_cart, basis, np.array(raw_orbs).T)
    #molc_orb = np.array(raw_orbs).T

    # if in spherically adapted orbitals, first convert
    # to cartesians
    if not in_cart:
        molc_orb_cart = moinfo.sph2cart(basis, molc_orb, sph2cart)
        nao_cart      = molc_orb_cart.shape[0]
    else:
        molc_orb_cart = molc_orb
        nao_cart      = n_aos

    # copy orbitals into gam_orb
    gam_orb = moinfo.Orbitals(nao_cart, n_mos)
    gam_orb.mo_vectors = molc_orb_cart
    gam_orb.occ = molc_occ

    # construct mapping array from TURBOMOLE to GAMESS
    molc_gam_map, scale_molc, scale_gam = basis.construct_map(ao_ordr, ao_norm)

    # remove the molcas normalization factors
    gam_orb.scale(scale_molc)

    # re-sort orbitals to GAMESS ordering
    gam_orb.sort_aos(molc_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb

def sort_molcas_orbs(in_cart, basis, orb_raw):
    """puts AOs in canonical order"""
  
    ang_max = 4 
    if in_cart:
        nfunc = moinfo.nfunc_cart
        aos   = ao_ordr
    else:
        nfunc = moinfo.nfunc_sph
        aos   = ao_ordr_sph
 
    (l_mo, n_mo) = orb_raw.shape
    std_array   = []
    molc_array  = []
    for iatom in range(len(basis.n_bf)):
        nbf     = len(basis.basis_funcs[iatom])
        ang_lst = [basis.basis_funcs[iatom][ibf].ang_mom for ibf in range(nbf)]
        n_ang   = [ang_lst.count(iang) for iang in range(ang_max)] 
        for iang in range(ang_max):
            for ifunc in range(n_ang[iang]):
                istr = str(iatom)+str(ifunc)
                std_array.extend([istr+aos[iang][i] 
                                  for i in range(len(aos[iang]))]) 
            for ifunc in range(nfunc[iang]):
                istr = str(iatom)
                molc_array.extend([istr+str(i)+aos[iang][ifunc] 
                                  for i in range(n_ang[iang])]) 
 
    map_array = [molc_array.index(std_array[i]) for i in range(n_mo)]
    orb_sort  = np.array([orb_raw[map_array,i_mo] for i_mo in range(n_mo)], dtype=float).T

    return orb_sort     

def generate_csf_list(ci_file):
    """This routine parses a molcas log file and pulls out the determinants
       assuming the keyword PRSD has been used. So: this function name
       is a misnomer. That should be fixed in the future. This routine
       parses an entire input file and pulls out all the determinant lists,
       if more than one RASSCF section is present, numerals "2", "3", etc.
       will be appended to the determinant list file names"""

    # number of 'frozen' orbitals (doubly occupied)
    # number of act = RAS1 + RAS2 + RAS3
    # unique list of determinants with coefficients
    parse_molcas_log(ci_file)

    return

def generate_csf_list(ci_file):
    """This routine parses a molcas log file and pulls out the determinants
       assuming the keyword PRSD has been used. So: this function name
       is a misnomer. That should be fixed in the future. This routine
       parses an entire input file and pulls out all the determinant lists,
       if more than one RASSCF section is present, numerals "2", "3", etc.
       will be appended to the determinant list file names"""

    dfile_suffix = [''] + ['_'+str(i) for i in range(10)]
    dfile_prefix = ['S','D','T','Q','5']

    ngrp         = 0
    scan_roots   = False
    det_list     = []
    f_prefix     = ''
    f_suffix     = ''
    nclsd        = 0
    nact         = 0
    ntot         = 0
    norb         = 0

    with open(ci_file, 'r') as molcas_log:
        for line in molcas_log:

            # we've come across a new RASSCF section
            if 'Number of closed shell electrons' in line:

                # print the current determinant list from
                # previous section
                if ntot > 0:
                    norb = max(norb, nclsd+nact)
                    print_det_list(f_prefix, f_suffix, nclsd+nact, det_list)

                # if there were any determinants in previous section
                # update the grp variable (changes file suffix)
                if sum([len(idets) for idets in det_list]) > 0:
                    ngrp    += 1
                    f_suffix = dfile_suffix[ngrp]

                nclsd     = int(0.5*int(line.split()[5]))
                clsd_orbs = [2] * nclsd
                det_list  = []
                iroot     = -1
                ntot      = 0

            if '      Active orbitals' in line:
                nact = int(line.split()[2])

            if 'Spin quantum number' in line:
                S = float(line.split()[3])
                f_prefix = dfile_prefix[int(2*S)]

            # we've come across a potential determinant list
            if 'printout of CI-coefficients larger than' in line:
                iroot     += 1
                scan_roots = True
                det_list.append([])
                
            # need to extract csf weight to determine det cf:
            if scan_roots and len(line.split()) == 4:
                try:
                    csf_cf = float(line.split()[2])
                except:
                    csf_cf = 0

            # if we hit a determinant, add it to the list
            if line[5:10] == 'sqrt(':
                line_lst  = line.split()
                sgn       = line_lst[0]
                num,denom = line_lst[2].split('/')
                cf        = csf_cf * np.sqrt( float(num) / float(denom))
                if sgn == '-':
                    cf *= -1.
                det       = list(line_lst[4].strip('|'))
                for orb, occ  in enumerate(det):
                    if occ == 'a':
                        det[orb] = 1
                    elif occ == 'b':
                        det[orb] = -1
                    else:
                        det[orb] = int(occ)
                det_full = clsd_orbs + det

                # if determinant is unique, add to list, else
                # update the coefficient on existing det
                found = False
                for idet in range(len(det_list[iroot])):
                    diff = [ai - bi for ai,bi in 
                            zip(det_full, det_list[iroot][idet][1:])]
                    if diff == [0]*(nclsd + nact):
                        #update det_list coeff
                        det_list[iroot][idet][0] += cf
                        found = True
                        break

                # else, extend the list
                if not found:
                    det_list[iroot].append([cf]+det_full)
                    ntot += 1      


            # finding those csf weights is difficult, as the
            # line formatting is not particularly unique. So:
            # when we're out of the root listing section,
            # stop looking for them
            if scan_roots and 'Natural orbitals and ' in line:
                scan_roots = False


    return norb


def print_det_list(f_prefix, f_suffix, norbs, det_list):
    """Prints det list in csf2det format."""

    det_fmt = ('{:14.10f}'+''.join('{:3d}' for i in range(norbs))+
                           '\n')

    nrm_file = open('norms.dat', 'x')

    for st in range(len(det_list)):
        fname    = f_prefix+str(st)+'_det'+f_suffix+'.dat'
        dat_file = open(fname, 'w')

        # sort the list
        cf_abs = [abs(idet[0]) for idet in det_list[st]]
        nrm    = np.sqrt(sum([icf**2 for icf in cf_abs]))
        ordr = np.flip(np.argsort(cf_abs))

        nrm_file.write(fname+': '+'{:10.5f}\n'.format(nrm))

        for idet in range(len(det_list[st])):
            det = det_list[st][ordr[idet]]
            dat_file.write(det_fmt.format(*det))

        dat_file.close()

    nrm_file.close()
    return

