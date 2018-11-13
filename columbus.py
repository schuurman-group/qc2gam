# 
# A set of routines to parse the output of a columbus 
# calculation
#
#

import sys
import moinfo

def parse(geom_file, basis_file, mo_file):
    """Documentation to come"""

    # read columbus geometry file
    gam_geom  = read_geom(geom_file)
    
    # parse daltaoin file to get basis information
    [in_cart, gam_basis] = read_basis(basis_file, gam_geom)

    # parse mocoef file to pull out orbitals
    gam_mos   = read_mos(in_cart, mo_file)

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

    # slurp up daltaoin file
    with open(basis_file) as daltaoin:
        bfile = daltaoin.readlines()
   
    # step through daltaoin file and read functions

    # first thing to check: cartesians/spherical and how many
    # groups of contractions
    i = 0
    while(bfile[i].split()[0].lower() != 'c' and 
               bfile[i].split()[0].lower() != 's'):
        i+=1

    in_cartestians = (bfile[i].split()[0].lower() == 'c'
    n_grp          = int(bfile[i].split()[1])

    # iterate over the number of groups
    i+=1
    atm_cnt = 0
    for j in range(n_grp):
 
        [a_num, n_atm, n_shl] = bfile[i].split()[0:3]
        n_per_shell           = bfile[i].split()[3:]
 
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
                [int_prog, n_prim, n_con] = bfile[i].split()
                b_funcs = [moinfo.basis_function(ang_mom) for m in range(n_con)]
                for m in range(n_prim):
                    i+=1
                    expon = float(bfile[i].split()[0])
                    coefs = bfile[i].split()[1:]
                    for n in range(n_con):
                        b_funcs[n].add_primitive(expon, coefs[n])
                for m in range(n_atm):
                    for n in range(n_con):
                        basis.add_function(atm_cnt+m, b_funcs[n])
                   
        # increment atom counter
        atm_cnt += n_atm

    return [in_cartesians, basis]

def read_mos(mocoef_file):
    """Documentation to come"""


    return



