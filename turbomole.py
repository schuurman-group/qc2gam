# 
# A set of routines to parse the output of a turbomole
# calculation
#
#

import sys
import numpy as np
import moinfo

# TURBOMOLE orbital ordering (in cartesians)
# s px py pz dxx dxy dxz dyy dyz dzz fxxx fxxy fxxz fxyy fxyz fxzz fyyy fyyz fyzz fzzz
ao_ordr         = [['s'],
                   ['px','py','pz'],
                   ['dxx','dxy','dxz','dyy','dyz','dzz'],
                   ['fxxx','fxxy','fxxz','fxyy','fxyz',
                    'fxzz','fyyy','fyyz','fyzz','fzzz']]

# dxy dxz dyz
ao_norm         = [[1.],
                   [1.,1.,1.],
                   [0.5,np.sqrt(3.),np.sqrt(3.),0.5,np.sqrt(3.),0.5],
                   [np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
                    np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
                    np.sqrt(15.),np.sqrt(15.)]]

#ao_norm         = [[1.],
#                   [1.,1.,1.],
#                   [1./np.sqrt(3.),1.,1.,1./np.sqrt(3.),1.,1./np.sqrt(3.),1.],
#                   [1.,1.,1.,1.,1.,1.,1.,1.,1.,.1]]



# how to convert from spherical to cartesian basis functions (in turbomole ordering)
# s -> s | px -> px | py -> py | pz -> pz 
# d ordering: d0, d1, d1-, d2-, d2
# dxx ->  -d0 + d2+ * sqrt(3)
# dxy ->  d2- 
# dxz ->  d1+
# dyy -> -d0 - d2+ * sqrt(3)
# dyz ->  d1- 
# dzz -> 2. * d0
# f ordering: f0, f1, f1-, f2-, f2, f3, f3-
# fxxx -> -f1+ - f3+/15.
# fxxy -> -f1- + sqrt(2)*f3-
# fxxz -> f0 + f1
# fxyy -> -f1 + f3
# fxyz -> f2-
# fxzz -> 4 * f1
# fyyy -> -f1- - sqrt(2)*f3-
# fyyz -> f0 - f2
# fyzz -> 4 * f1-
# fzzz -> -2 * f0 / 3
sph2cart        = [
                   [ [[0],[1.]] ],                                     # conversion for s orbitals
                   [ [[0],[1.]], [[1],[1.]], [[2],[1.]] ],             # conversion for p orbitals
                   [ [[0,4],[-1.,np.sqrt(3.)]], 
                     [[3],[1.]], 
                     [[1],[1.]],          # conversion for d orbitals
                     [[0,4],[-1.,-np.sqrt(3.)]], 
                     [[2],[1.]], 
                     [[0],[2.]] ],
                   [ [[1,5],[-1.,-1./15.]], 
                     [[2,6],[-1., np.sqrt(2.)]], # conversion for f orbitals
                     [[0,1],[1.,1.]], 
                     [[2,5],[-1.,1.]], 
                     [[3],[1.]], 
                     [[1],[4.]], 
                     [[2,6],[-1, -np.sqrt(2.)]], 
                     [[0,4],[1.,-1.]],
                     [[2],[4.]], 
                     [[0], [-2./3.]] ]
                  ] 

def parse(geom_file, basis_file, mo_file):
    """Documentation to come"""

    # read turbomole coord file
    gam_geom  = read_geom(geom_file)
    
    # parse basis file to get basis set information
    [in_cart, gam_basis] = read_basis(basis_file, gam_geom)

    # parse mos file to pull out orbitals
    gam_mos   = read_mos(mo_file, in_cart, gam_basis)

    return [gam_basis, gam_mos]

#
def read_geom(geom_file):
    """reads a turbomole coord file and loads results into moinfo.geom format"""

    # create geometry object to hold the data
    geom = moinfo.geom()

    # slurp up geometry file
    with open(geom_file) as turbo_coord: 
        gfile = turbo_coord.readlines()

    i = 0
    while(gfile[i].split()[0] != "$coord"):
        i+=1
   
    atoms = []
    i+=1
    while('$' not in gfile[i]):
        gstr   = gfile[i].split() # format = coords, asym
        asym   = gstr[3].upper().rjust(2)
        coords = [float(gstr[i])*moinfo.au2ang for i in range(3)]
        geom.add_atom(moinfo.atom(asym,coords))
        i      += 1

    return geom

def read_basis(basis_file, geom):
    """Documentation to come"""

    # create basis set object
    basis = moinfo.basis_set('unknown',geom)

    # slurp up daltaoin file
    with open(basis_file) as turbo_basis:
        bfile = turbo_basis.readlines()
   
    # default is to run in spherically adapted functions
    # can add for cartesians later (not sure how to check
    # that at the moment)
    in_cartesians = False

    # look for start of basis section 
    i = 0
    while(bfile[i].split()[0] != "$basis"):
        i += 1

    # iterate over the atom types
    b_set     = ""
    a_sym     = ""
    sec_start = False
    i         += 1
    while True:
        line = bfile[i].split()

        # stepping on $end line breaks us out of parser
        if line[0] == "$end":
            break

        # ignore blank lines
        if len(line) == 0:
            i += 1
            continue

        # ignore comment lines
        if line[0][0] == "#":
            i += 1
            continue

        # if first string is star, either
        # beginning or ending basis section
        if line[0] == "*":
            sec_start = not sec_start
            i += 1
            continue

        # if starting section, first line 
        # is atom and basis set line
        if sec_start:
            a_sym = line[0]
            b_set = line[1]
            a_lst = [ind for ind,sym in enumerate([geom.atoms[k].symbol 
                       for k in range(geom.natoms())]) 
                       if sym == a_sym.upper().rjust(2)]
            i += 1
            continue

        # if we get this far, we're parsing the basis set!
        [nprim, ang_sym] = line
        ang_mom          = moinfo.ang_mom_sym.index(ang_sym.upper())
        bfunc            = moinfo.basis_function(ang_mom)
        for j in range(int(nprim)):
            i += 1
            [exp, coef]  = bfile[i].split()
            bfunc.add_primitive(float(exp), float(coef))
        for atom in a_lst:
            basis.add_function(atom, bfunc) 
        i += 1

    return [in_cartesians, basis]

def read_mos(mocoef_file, in_cart, basis):
    """Documentation to come"""

    # So: in the future, we should also allow for a mapping array that
    # allows for taking linear combinations of elements in order to convert
    # from a spherically adapted basis to cartesians. That will come later
    # though...first the easy stuff.    

    # slurp up the mocoef file
    with open(mocoef_file) as mos:
        mo_file = mos.readlines()

    i        = 0
    wid      = 20
    n_mos    = 0
    n_ao_all = []
    raw_orbs = []
    new_orb = False
    while i<len(mo_file):
        line = mo_file[i].split()

        # ignore the first line
        if line[0] == "$scfmo" or line[0] == "$end":
            i += 1
            continue

        # if comment line, then ignore
        if line[0] == '#':
            i += 1
            continue
    
        # if line contains "eigenvalue", start new orbitals   
        if 'eigenvalue' in ' '.join(line).lower():
            raw_orbs.append([])
            n_ao_all.extend([0])
            n_mos   += 1
            new_orb = True
            i       += 1
            continue
 
        # else, parse a line
        mo_row = line[0].lower().replace('d','e') 
        n_cf = int(len(mo_row)/wid)
        for j in range(n_cf):
            n_ao_all[-1] += 1  
            raw_orbs[-1].extend([float(mo_row[j*wid:(j+1)*wid])])
        i += 1

    # make sure all the mos are the same length
    uniq = list(set(n_ao_all))
    if len(uniq) != 1:
        sys.exit('MOs have different numbers of AOs. Exiting...')
    else:
        n_aos = uniq[0]    

    # determine if we're running in cartesian or spherically adapted
    # atomic basis functions
    if n_aos == basis.n_bf_cart:
        in_cart = True
    elif n_aos == basis.n_bf_sph:
        in_cart = False
    else:
        sys.exit('Number of basis functions is not equal to: '+str(uniq[0]))

    # if in spherically adapted orbitals, first convert
    # to cartesians
    turb_orb = np.array(raw_orbs).T
    if not in_cart:
        turb_orb_cart = moinfo.sph2cart(basis, turb_orb, sph2cart)
        nao_cart      = turb_orb_cart.shape[0]
    else:
        turb_orb_cart = turb_orb
        nao_cart      = n_aos

    # copy orbitals into gam_orb 
    gam_orb = moinfo.orbitals(nao_cart, n_mos)
    gam_orb.mo_vectors = turb_orb_cart

    # construct mapping array from TURBOMOLE to GAMESS
    [turb_gam_map, scale_turb, scale_gam] = basis.construct_map(ao_ordr, ao_norm)

    # remove the dalton normalization factors
    gam_orb.scale(scale_turb)
  
    # re-sort orbitals to GAMESS ordering
    gam_orb.sort(turb_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb



