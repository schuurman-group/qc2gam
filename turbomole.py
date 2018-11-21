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

ao_norm         = [[1.],
                   [1.,1.,1.],
                   [np.sqrt(3.),np.sqrt(3.),np.sqrt(3.),np.sqrt(3.),
                    np.sqrt(3.),np.sqrt(3.)],
                   [np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
                    np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
                    np.sqrt(15.),np.sqrt(15.)]]

# how to convert from spherical to cartesian basis functions (in turbomole ordering)
# s -> s | px -> px | py -> py | pz -> pz 
# d ordering: d0, d1, d1-, d2-, d2
# dxx ->  -d0 + d2+
# dxy ->  d2-
# dxz ->  d1+
# dyy -> -d0 - d2+
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
                   [ [[0,4],[-1.,1]], 
                     [[3],[1.]], 
                     [[1],[1.]],          # conversion for d orbitals
                     [[0,4],[-1.,-1.]], 
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
    while(gfile[i].split()[0] != "$coord")
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

    # initialize the ordering array
    turbo_gam_map = [[] for i in range(geom.natoms())]

    # slurp up daltaoin file
    with open(basis_file) as turbo_basis:
        bfile = turbo_basis.readlines()
   
    # default is to run in spherically adapted functions
    # can add for cartesians later (not sure how to check
    # that at the moment)
    in_cartesians = False

    # iterate over the number of groups
    i = 0
    atm_cnt = 0
    for j in range(n_grp):
        i += 1
        [n_atm,n_shl]         = list(map(int,bfile[i].split()[1:3]))
        n_per_shell           = list(map(int,bfile[i].split()[3:]))
 
        # for time being assume C1, so number of atoms in
        # daltaoin file == number of atoms in geom file
        for k in range(n_atm):
          i+=1
          [a_sym, a_index, x, y, z, sym] = bfile[i].split()
          if a_sym.upper().rjust(2) != geom.atoms[atm_cnt+k].symbol:
              sys.exit('Mismatch between daltaoin and geom file, atom='+str(atm_cnt+k))

        # loop over the number of angular momentum shells
        ang_mom = -1
        for k in range(n_shl):
            ang_mom += 1
            for l in range(n_per_shell[k]):
                i+=1 
                [n_prim, n_con] = list(map(int,bfile[i].split()[1:]))
                b_funcs = [moinfo.basis_function(ang_mom) for m in range(n_con)]
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

    # move to the first line of the orbital coefficients, reading nao and nmo
    # along the way
    line_index = 0
    while(mo_file[line_index].split()[0][0] != '('):
        line_index += 1
        # figure out nao, nmo
        if mo_file[line_index].split()[0] == "A":
            [nao, nmo] = list(map(int,mo_file[line_index-1].split()))

    # create a numpy array to hold orbitals
    col_orb = np.zeros((nao,nmo),dtype=float)
    for i in range(nmo):
        n_remain = nao
        row      = 0
        while(n_remain > 0):
            line_index += 1
            line_arr    = mo_file[line_index].split()
            col_orb[3*row:3*row+min(n_remain,3),i] = np.array(
               [line_arr[k].replace("D","e") for k in range(len(line_arr))],dtype=float)             
            n_remain -= min(n_remain,3)
            row += 1

    # make the map array
    nf_cnt = 0
    dalt_gam_map = []
    scale_col     = []
    scale_gam     = []

    if not in_cart:
        nfunc_nascent = moinfo.nfunc_sph
    else:
        nfunc_nascent = moinfo.nfunc_cart
    ang_mom_ao = []

    for i in range(basis.geom.natoms()):
        for j in range(len(basis.basis_funcs[i])):
            ang_mom = basis.basis_funcs[i][j].ang_mom
            # if we have to eventually unroll these spherically adapted functions
            # make a note of where they are
            ang_mom_ao.extend([ang_mom for i in range(nfunc_nascent[ang_mom])])
            nfunc   = moinfo.nfunc_cart[ang_mom]            
            map_arr = [nf_cnt + ao_ordr[ang_mom].index(moinfo.ao_ordr[ang_mom][k]) 
                       for k in range(nfunc)]
            dalt_gam_map.extend(map_arr)
            scale_col.extend([ao_norm[ang_mom][k] for k in range(nfunc)])
            scale_gam.extend([1./moinfo.ao_norm[ang_mom][k] for k in range(nfunc)])
            nf_cnt += nfunc

    # if in spherically adapted orbitals, first convert
    # to cartesians
    if not in_cart:
        orb_trans = [[] for i in range(nmo)]
        ang_mom_max = max(ang_mom_ao)
        iao = 0
        iao_cart = 0
        while(iao<nao):
            # only an issue for l>=2 
            if ang_mom_ao[iao]>=2:
                lval = ang_mom_ao[iao]
                for imo in range(nmo):
                    cart_orb = [sum([col_orb[iao+
                                sph2cart[lval][j][0][k],imo]*sph2cart[lval][j][1][k] 
                                for k in range(len(sph2cart[lval][j][0]))])
                                for j in range(moinfo.nfunc_cart[lval])]
                    orb_trans[imo].extend(cart_orb)
                iao      += nfunc_nascent[lval]
                iao_cart += moinfo.nfunc_cart[lval] 
            else:
                for imo in range(nmo):
                    orb_trans[imo].extend([col_orb[iao,imo]])
                iao      += 1
                iao_cart += 1

        col_orb_cart = np.array(orb_trans).T
        nao_cart     = iao_cart
    else:
        col_orb_cart = col_orb
        nao_cart     = nao 
      
    # copy orbitals into gam_orb 
    gam_orb = moinfo.orbitals(nao_cart, nmo)
    gam_orb.mo_vectors = col_orb_cart

    # remove the dalton normalization factors
    gam_orb.scale(scale_col)
  
    # re-sort orbitals to GAMESS ordering
    gam_orb.sort(dalt_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb



