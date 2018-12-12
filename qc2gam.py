#!/usr/bin/env python
"""
Program to convert quantum chemistry output
(i.e. CI wavefunctions and molecular orbitals)
to GAMESS/Multigrid type format
"""
import sys
import columbus
import dalton
import turbomole


def process_arguments(args):
    """Process command line arguments."""
    input_style = None
    geom_file   = None
    gorder      = None
    mo_file     = None
    basis_file  = None
    ci_file     = None
    out_file    = None

    # read the command line arguments
    if '-input' in sys.argv:
        input_style = args[args.index('-input')+1]

    if '-geom' in sys.argv:
        geom_file = args[args.index('-geom')+1]

    if '-ordr' in sys.argv:
        gorder = args[args.index('-ordr')+1]

    if '-mos' in sys.argv:
        mo_file  = args[args.index('-mos')+1]

    if '-basis' in sys.argv:
        basis_file = args[args.index('-basis')+1]

    if '-ci' in sys.argv:
        ci_file    = args[args.index('-ci')+1]

    if '-output' in sys.argv:
        out_file   = args[args.index('-output')+1]

    # use the default output file name (mos.dat) if this has not been given
    # by the user
    if out_file == None:
        out_file = 'mos.dat'

    # if they have not been given, fill in the names of the geometry,
    # MO and basis files using the names usually used by the given program
    if input_style == 'turbomole':
        if geom_file == None:
            geom_file = 'coord'
        if mo_file == None:
            mo_file = 'mos'
        if basis_file == None:
            basis_file = 'basis'
    elif input_style == 'columbus':
        if geom_file == None:
            geom_file = 'geom'
        if mo_file == None:
            mo_file = 'mocoef'
        if basis_file == None:
            basis_file = 'daltaoin'
    else:
        raise ValueError('input style '+str(inp)+' not recognized.')

    return input_style, geom_file, gorder, basis_file, mo_file, ci_file, out_file


def convert(inp, gm, go, basis, mos, ci, out):
    """Convert orbital/basis information to GAMESS format."""
    # import the appropriate module
    qc_input =__import__(inp, fromlist=['a'])

    if ci is not None:
        qc_input.generate_csf_list(ci)

    # parse the output and convert to GAMESS format
    gam_basis, gam_mos = qc_input.parse(gm, go, basis, mos)

    # print the MOs file
    gam_basis.print_basis_set(out)
    gam_mos.print_orbitals(out)


if __name__ == '__main__':
    # parse command line arguments
    args = process_arguments(sys.argv)

    # run the program
    convert(*args)

