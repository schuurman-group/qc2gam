"""
A set of routines to parse the output of a columbus
calculation
"""
import sys
import numpy as np
import moinfo


# DALTON orbital ordering (in cartesians)
# s px py pz dxx dxy dxz dyy dyz dzz
# fxxx fxxy fxxz fxyy fxyz fxzz fyyy fyyz fyzz fzzz
ao_ordr = [['s'],
           ['px', 'py', 'pz'],
           ['dxx', 'dxy', 'dxz', 'dyy', 'dyz', 'dzz'],
           ['fxxx', 'fxxy', 'fxxz', 'fxyy', 'fxyz',
            'fxzz', 'fyyy', 'fyyz', 'fyzz', 'fzzz']]

ao_norm = [[1.],
           [1.,1.,1.],
           [np.sqrt(3.),np.sqrt(3.),np.sqrt(3.),np.sqrt(3.),
            np.sqrt(3.),np.sqrt(3.)],
           [np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
            np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
            np.sqrt(15.),np.sqrt(15.)]]

# how to convert from spherical to cartesian basis functions
# (in columbus ordering)
# s -> s | px -> px | py -> py | pz -> pz
# d ordering: d2-, d1-, d0, d1+ ,d2+
# dxx ->  -d0 + d2+
# dxy ->  d2-
# dxz ->  d1+
# dyy -> -d0 - d2+
# dyz ->  d1-
# dzz -> 2. * d0
# f ordering: f3-, f2-, f1-, f0, f1+, f2+, f3+
# fxxx -> -f1+ - f3+/15.
# fxxy -> sqrt(2)*f3- - f1-
# fxxz -> f0 + f1
# fxyy -> -f1 + f3
# fxyz -> f2-
# fxzz -> 4 * f1
# fyyy -> -sqrt(2)*f3- - f1-
# fyyz -> f0 - f2
# fyzz -> 4 * f1-
# fzzz -> -2 * f0 / 3
sph2cart = [
    [[[0], [1.]]],                           # conversion for s orbitals
    [[[0], [1.]], [[1], [1.]], [[2], [1.]]], # conversion for p orbitals
    [[[2, 4], [-1., 1]],                     # conversion for d orbitals
     [[0], [1.]],
     [[3], [1.]],
     [[2, 4], [-1., -1.]],
     [[1], [1.]],
     [[2], [2.]]],
    [[[4, 6], [-1., -1./15.]],               # conversion for f orbitals
     [[0, 2], [np.sqrt(2.), -1.]],
     [[3, 5], [1., 1.]],
     [[4, 6], [-1., 1.]],
     [[1], [1.]],
     [[4], [4.]],
     [[0, 2], [-np.sqrt(2.), -1.]],
     [[3, 5], [1., -1.]],
     [[2], [4.]],
     [[3], [-2./3.]]]
            ]


def parse(geom_file, geom_ordr, basis_file, mo_file):
    """Parses a set of columbus input files."""
    # read columbus geometry file
    gam_geom = read_geom(geom_file)
    if geom_ordr is not None:
        gam_ordr = read_geom(geom_ordr)
        gam_geom.reorder(gam_ordr)

    # parse daltaoin file to get basis information
    in_cart, gam_basis = read_basis(basis_file, gam_geom)

    # parse mocoef file to pull out orbitals
    gam_mos = read_mos(mo_file, in_cart, gam_basis)

    return gam_basis, gam_mos


def read_geom(geom_file):
    """Reads a columbus geom and loads results into moinfo.Geom."""
    # create geometry object to hold the data
    geom = moinfo.Geom()

    # slurp up geometry file
    with open(geom_file, 'r') as col_geom:
        gfile = col_geom.readlines()

    atoms = []
    for i in range(len(gfile)):
        gstr   = gfile[i].split() # format = asym, anum, coords
        asym   = gstr[0].upper().rjust(2)
        coords = [float(gstr[i])*moinfo.au2ang for i in range(2,5)]
        atom_i = moinfo.Atom(asym, coords)
        geom.add_atom(atom_i)

    return geom


def read_basis(basis_file, geom):
    """Reads a columbus basis file into moinfo.BasisSet."""
    # create basis set object
    basis = moinfo.BasisSet('unknown', geom)

    # slurp up daltaoin file
    with open(basis_file, 'r') as daltaoin:
        bfile = daltaoin.readlines()

    # step through daltaoin file and read functions
    # first thing to check: cartesians/spherical and how many
    # groups of contractions
    for i in range(len(bfile)):
        if len(bfile[i].split()) > 0:
            if (bfile[i].split()[0].lower() == 'c' or
                bfile[i].split()[0].lower() == 's'):
                break

    in_cartesians = bfile[i].split()[0].lower() == 'c'
    n_grp         = int(bfile[i].split()[1])

    # iterate over the number of groups
    atm_cnt = 0
    for j in range(n_grp):
        i += 1
        n_atm, n_shl = list(map(int,bfile[i].split()[1:3]))
        n_per_shell  = list(map(int,bfile[i].split()[3:]))

        # for time being assume C1, so number of atoms in
        # daltaoin file == number of atoms in geom file
        for k in range(n_atm):
          i+=1
          a_sym, a_index, x, y, z, sym = bfile[i].split()
          if a_sym.upper().rjust(2) != geom.atoms[atm_cnt+k].symbol:
              raise ValueError('Mismatch between daltaoin and geom file, ' +
                               'atom='+str(atm_cnt+k))

        # loop over the number of angular momentum shells
        ang_mom = -1
        for k in range(n_shl):
            ang_mom += 1
            for l in range(n_per_shell[k]):
                i+=1
                n_prim, n_con = list(map(int,bfile[i].split()[1:]))
                b_funcs = [moinfo.BasisFunction(ang_mom) for m in range(n_con)]
                for m in range(n_prim):
                    i+=1
                    exp_con = list(map(float,bfile[i].split()))
                    expon = exp_con[0]
                    coefs = exp_con[1:]
                    for n in range(n_con):
                        b_funcs[n].add_primitive(expon, coefs[n])
                for m in range(n_atm):
                    for n in range(n_con):
                        basis.add_function(atm_cnt+m, b_funcs[n])

        # increment atom counter
        atm_cnt += n_atm

    return in_cartesians, basis


def read_mos(mocoef_file, in_cart, basis):
    """Reads the columbus molecular orbital file into moinfo.Orbitals.

    So: in the future, we should also allow for a mapping array that
    allows for taking linear combinations of elements in order to convert
    from a spherically adapted basis to cartesians. That will come later
    though... first the easy stuff.
    """
    # slurp up the mocoef file
    with open(mocoef_file, 'r') as mocoef:
        mo_file = mocoef.readlines()

    # move to the first line of the orbital coefficients, reading nao and nmo
    # along the way
    line_index = 0
    while(mo_file[line_index].split()[0][0] != '('):
        line_index += 1
        # figure out nao, nmo
        if mo_file[line_index].split()[0].lower() == 'a':
            nao, nmo = list(map(int, mo_file[line_index-1].split()))

    # create a numpy array to hold orbitals
    col_orb = np.zeros((nao, nmo), dtype=float)
    for i in range(nmo):
        n_remain = nao
        row      = 0
        while(n_remain > 0):
            line_index += 1
            line_arr    = mo_file[line_index].split()
            col_orb[3*row:3*row+min(n_remain,3),i] = np.array(
               [line_arr[k].replace('D','e') for k in range(len(line_arr))],
               dtype=float)
            n_remain -= min(n_remain, 3)
            row += 1

    # create a numpy array to hold populations, if present
    col_occ = np.zeros(nmo, dtype=float)
    nocc = 0
    for i in range(2, len(mo_file)):
        if 'orbocc' in mo_file[i-2] or nocc > 0:
            occ_line = np.array(mo_file[i].replace('D','e').split(), dtype=float)
            nrow = len(occ_line)
            col_occ[nocc:nocc+nrow] = occ_line
            nocc += nrow
    if nocc == 0:
        col_occ = None

    # if in spherically adapted orbitals, first convert
    # to cartesians
    if not in_cart:
        col_orb_cart = moinfo.sph2cart(basis, col_orb, sph2cart)
        nao_cart     = col_orb_cart.shape[0]
    else:
        col_orb_cart = col_orb
        nao_cart     = nao

    # copy orbitals into gam_orb
    gam_orb = moinfo.Orbitals(nao_cart, nmo)
    gam_orb.mo_vectors = col_orb_cart
    gam_orb.occ = col_occ

    # construct mapping array from COLUMBUS to GAMESS
    dalt_gam_map, scale_col, scale_gam = basis.construct_map(ao_ordr, ao_norm)

    # remove the dalton normalization factors
    gam_orb.scale(scale_col)

    # re-sort orbitals to GAMESS ordering
    gam_orb.sort(dalt_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb


def generate_csf_list(ci_file):
    """Generates a list of CSFs from a cipcls file."""
    valid, is_cipc = is_cipcls(ci_file)

    if not valid:
        raise SyntaxError('Cannot parse csf list output: not cipcls or mcpcls')

    # parse cipcls or mcpcls file
    n_occ, n_extl, csf_list = parse_ci_file(ci_file, is_cipc)

    # print the csf_list to file
    print_csf_list(n_occ, n_extl, csf_list)


def is_cipcls(in_file):
    """Returns true, true if the file to parse is a cipcls file,
    true, false if mcpcls file, and false, false is file not
    recognized."""
    ci_str = 'PROGRAM:              CIPC'
    mc_str = 'PROGRAM:              MCPC'

    with open(in_file, 'r') as ci_file:
        for line in ci_file:
            # if cipcls file, return valid_file=True, cipcls=True
            if ci_str in line:
                return True, True

            # if mcpcls file, return valid_file=True, cipcls=False
            if mc_str in line:
                return True, False

    # if neither cipcls or mcpcls file, file not valid
    return False, False


def parse_ci_file(ci_file, is_cipc):
    """Parses cipcls file, extracts csf list and coefficients."""
    csf_list   = []
    ci_str     = '  ------- -------- ------- - ---- --- ---- --- ------------'
    mc_str     = '-----  ------------  ------------  ------------'
    ndocc      = 0
    nocc       = 0
    nextl      = 0
    parse_line = False
    read_docc  = False
    istate     = -1
    with open(ci_file, 'r') as cipcls:
        for line in cipcls:
            # read the number of frozen orbitals (if cipcls)
            if 'frozen orbital =' in line and is_cipc:
                l_arr = line.strip().split()
                ndocc = len(l_arr)-3
                continue

            # read the number of docc orbitals (if mcpcls)
            if 'List of doubly occupied orbitals' in line and not is_cipc:
                read_docc = True
                continue

            # read in the number of doubly occupied orbitals (for mcpcls)
            if read_docc:
                l_arr = line.strip().split()
                if len(l_arr) == 0:
                    read_docc = False
                    continue
                else:
                    ndocc += len(l_arr)/2
                    continue

            # about to start reading csf list:
            if ci_str in line or mc_str in line:
                istate += 1
                csf_list.append([])
                parse_line = True
                continue

            # read a csf line
            if parse_line:
                l_arr = line.strip().split()

                # stopping criteria for cipcls
                if 'csfs were printed' in line:
                    parse_line = False
                    continue

                # stopping crieria for mcpcls
                if len(l_arr) == 0:
                    parse_line = False
                    continue

                n_int, n_ext, csf_vec = parse_ci_line(is_cipc, ndocc, l_arr)
                csf_list[istate].append([float(l_arr[1]),csf_vec])

                # total number of external orbitals is set to the csfs with
                # highest excitation order
                nextl = max(nextl, n_ext)

    return n_int, nextl, csf_list


def print_csf_list(n_occ, n_extl, csf_list):
    """Prints csf list in csf2det format."""
    csf_fmt = ('{:14.10f}'+''.join('{:2d}' for i in range(n_occ))+
                           ''.join('{:4d}:{:2d}' for i in range(n_extl))+
                           '\n')

    print("n_occ="+str(n_occ)+"\n")
    for state in range(len(csf_list)):
        dat_file = open('csf'+str(state+1), 'x')

        for csf in csf_list[state]:
            data = [csf[0]]
            ext_vec = [0] * (2*n_extl)
            n_ext = int((len(csf[1]) -  n_occ) / 2)
            print("n_extl, csf, n_ext="+str(n_extl)+" / "+str(csf)+" / "+str(n_ext)+"\n")
            for i in range(n_ext):
                ext_vec[2*i]   = csf[1][n_occ+n_ext+i]
                ext_vec[2*i+1] = csf[1][n_occ+i]

            data.extend(csf[1][0:n_occ])
            data.extend(ext_vec)
            dat_file.write(csf_fmt.format(*data))

        dat_file.close()


def parse_ci_line(is_cipc, ndocc, l_arr):
    """Parses an occupation array from cipcls or mcpcls."""
    csf_vec = [3] * ndocc

    if is_cipc:
       nextl   = int((len(l_arr) - 5)/3)
       str_ind = 4 + 3*nextl
       nact    = len(l_arr[str_ind])-nextl
       nintl   = ndocc + nact

       # first add internal orbitals
       csf_vec.extend([int(l_arr[str_ind][i+nextl]) for i in range(nact)])

       # now add external
       csf_vec.extend([0] * (2*nextl))
       for i in range(nextl):
           csf_vec[nintl+i]       = int(l_arr[str_ind][i])
           csf_vec[nintl+nextl+i] = int(l_arr[3*(i+2)])

    else:
       csf_vec.extend([int(l_arr[3][i]) for i in range(len(l_arr[3]))])

    return nintl, nextl, csf_vec
