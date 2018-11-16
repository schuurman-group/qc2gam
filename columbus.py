# 
# A set of routines to parse the output of a columbus 
# calculation
#
#

import sys
import numpy as np
import moinfo

# DALTON orbital ordering (in cartesians)
# s px py pz dxx dxy dxz dyy dyz dzz fxxx fxxy fxxz fxyy fxyz fxzz fyyy fyyz fyzz fzzz
# DALTON orbital ordering (in spherical harmonics)
# s px py pz d2- d1- d0 d1+ d2+ f3- f2- f1- f0 f1+ f2+ f3+
ao_ordr         = [['s'],
                   ['px','py','pz'],
                   ['dxx','dxy','dxz','dyy','dyz','dzz'],
                   ['fxxx','fxxy','fxxz','fxyy','fxyz',
                    'fxzz','fyyy','fyyz','fyzz','fzzz']]

ao_norm         = [[1.],
                   [1., 1., 1.],
                   [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.), 
                    1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)],
                   [1./np.sqrt(15.),np.sqrt(5./3.),np.sqrt(5./3.),
                    np.sqrt(5./3.),np.sqrt(15.),1/np.sqrt(5./3.),
                    1./np.sqrt(15.),np.sqrt(5./3.),np.sqrt(5./3.),
                    1./np.sqrt(15.)]]

#
def parse(geom_file, basis_file, mo_file):
    """Documentation to come"""

    # read columbus geometry file
    gam_geom  = read_geom(geom_file)
    
    # parse daltaoin file to get basis information
    [in_cart, gam_basis] = read_basis(basis_file, gam_geom)

    # parse mocoef file to pull out orbitals
    gam_mos   = read_mos(mo_file, in_cart, gam_basis)

    return [gam_basis, gam_mos]

#
def read_geom(geom_file):
    """reads a columbus geom and loads results into moinfo.geom format"""

    # create geometry object to hold the data
    geom = moinfo.geom()

    # slurp up geometry file
    with open(geom_file) as col_geom: 
        gfile = col_geom.readlines()
   
    atoms = []
    for i in range(len(gfile)):
        gstr   = gfile[i].split() # format = asym, anum, coords
        asym   = gstr[0].upper().rjust(2)
        coords = [float(gstr[i]) for i in range(2,5)]
        atom_i = moinfo.atom(asym,coords)
        geom.add_atom(atom_i) 

    return geom

def read_basis(basis_file, geom):
    """Documentation to come"""

    # create basis set object
    basis = moinfo.basis_set('unknown',geom)

    # initialize the ordering array
    dalt_gam_map = [[] for i in range(geom.natoms())]

    # slurp up daltaoin file
    with open(basis_file) as daltaoin:
        bfile = daltaoin.readlines()
   
    # step through daltaoin file and read functions
    # first thing to check: cartesians/spherical and how many
    # groups of contractions
    i = 0
    while True:
        if len(bfile[i].split())>0:
            if (bfile[i].split()[0].lower() == 'c' or
               bfile[i].split()[0].lower() == 's'):
                break   

        i+=1

    in_cartesians = bfile[i].split()[0].lower() == 'c'
    n_grp         = int(bfile[i].split()[1])

    # iterate over the number of groups
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
    with open(mocoef_file) as mocoef:
        mo_file = mocoef.readlines()

    # move to the first line of the orbital coefficients, reading nao and nmo
    # along the way
    line_index = 0
    while(mo_file[line_index].split()[0][0] != '('):
        line_index += 1
        # figure out nao, nmo
        if mo_file[line_index].split()[0] == "A":
            [nao, nmo] = list(map(int,mo_file[line_index-1].split()))

    # create an orbital object to hold orbitals
    gam_orb = moinfo.orbitals(nao, nmo)
   
    for i in range(nmo):
        mo_vec   = np.zeros(nao, dtype=float)
        n_remain = nao
        row      = 0
        while(n_remain > 0):
            line_index += 1
            line_arr    = mo_file[line_index].split()
            mo_vec[3*row:3*row+min(n_remain,3)] = np.array(
               [line_arr[k].replace("D","e") for k in range(len(line_arr))],dtype=float)             
            n_remain -= min(n_remain,3)
            row += 1
        gam_orb.add(mo_vec)

    # make the map array
    nf_cnt = 0
    dalt_gam_map = []
    scale_col     = []
    scale_gam     = []
    for i in range(basis.geom.natoms()):
        for j in range(len(basis.basis_funcs[i])):
            ang_mom = basis.basis_funcs[i][j].ang_mom
            nfunc   = moinfo.nfunc_per_shell[ang_mom]
            map_arr = [moinfo.ao_ordr[ang_mom].index(ao_ordr[ang_mom][k]) for k in range(nfunc)]
            dalt_gam_map.extend([nf_cnt + map_arr[k] for k in range(len(map_arr))])
            scale_col.extend([1./ao_norm[ang_mom][k] for k in range(nfunc)])
            scale_gam.extend([moinfo.ao_norm[ang_mom][k] for k in range(nfunc)])
            nf_cnt += nfunc

    # remove the dalton normalization factors
    gam_orb.scale(scale_col)
  
    # re-sort orbitals to GAMESS ordering
    gam_orb.sort(dalt_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb



