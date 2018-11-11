# 
# A set of routines to parse the output of a columbus 
# calculation
#
#

import os
import moinfo

def parse(geom_file, basis_file, mo_file):
    """Documentation to come"""

    # read columbus geometry file
    gam_geom  = read_geom(geom_file)
    
    # parse daltaoin file to get basis information
    gam_basis = read_basis(basis_file, gam_geom)

    # parse mocoef file to pull out orbitals
    gam_mos   = read_mos(mo_file)

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
        bfile = daltaoin.readline()
   
    # step through daltaoin file and read functions
    # also take note if calculation uses cartesian functions
    for i in range(len(bfile)):
        bline = bfile[i].split()
        


    return

def read_mos(mocoef_file):
    """Documentation to come"""


    return



