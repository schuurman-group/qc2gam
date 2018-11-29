#!/usr/bin/env python
#
#
# Program to convert quantum chemistry output 
# (i.e. CI wavefunctions and molecular orbitals)
# to GAMESS/Multigrid type format
#
#

import sys
import columbus
import dalton
import turbomole

#
#
#
def process_arguments(args):
    """Process command line arguments."""

    if '-input' in sys.argv:
        input_style = args[args.index('-input')+1]
    else:
    # default to COLUMBUS
        input_style = 'columbus'

    if '-geom' in sys.argv:
        geom_file = args[args.index('-geom')+1]
    # default to geometry file: geom
    else:
        geom_file = 'geom'

    if '-mos' in sys.argv:
        mo_file  = args[args.index('-mos')+1]
    # default to mos file 'mocoef'
    else:
        mo_file = 'mocoef'

    if '-basis' in sys.argv:
        basis_file = args[args.index('-basis')+1] 
    # default to daltaoin
    else:
        basis_file = 'daltaoin'

    if '-ci' in sys.argv:
        ci_file    = args[args.index('-ci')+1]
    # default to None
    else:
        ci_file    = None

    if '-output' in sys.argv:
        out_file   = args[args.index('-output')+1]
    else:
        out_file   = 'mos.dat'

    return [input_style, geom_file, basis_file, mo_file, ci_file, out_file]

#
#
#
def convert(inp, gm, basis, mos, ci, out):
    """Documentation to come"""
    input_styles = ['columbus', 'turbomole']
    
    try:
        input_mode = input_styles.index(inp)
    except:
        sys.exit('input style '+str(inp)+' not recognized.')

    # import the appropriate module
    qc_input =__import__(input_styles[input_mode], fromlist=['a'])
    
    if ci is not None:
        qc_input.generate_csf_list(ci)

    # parse the output and convert to GAMESS format
    [gam_basis, gam_mos] = qc_input.parse(gm, basis, mos)

    # print the MOs file
    gam_basis.print_basis_set(out)
    gam_mos.print_orbitals(out)


if __name__ == '__main__':

    # parse command line arguments
    [inp, gm, basis, mos, ci, out] = process_arguments(sys.argv)

    # run the program
    convert(inp, gm, basis, mos, ci, out)

