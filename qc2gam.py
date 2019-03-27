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


def get_arg(kwd, default):
    """Gets a keyword argument from sys.argv if present, otherwise sets
    the value to a given default."""
    args = sys.argv
    if kwd in args:
        return args[args.index(kwd)+1]
    else:
        return default


def process_arguments():
    """Process command line arguments."""
    # if they have not been given, fill in the names of the output, geometry,
    # MO and basis files using the names usually used by the given program
    input_style = get_arg('-input', None)
    if input_style == 'turbomole':
        geom_file = 'coord'
        mo_file = 'mos'
        basis_file = 'basis'
    elif input_style == 'columbus':
        geom_file = 'geom'
        mo_file = 'mocoef'
        basis_file = 'daltaoin'
    else:
        raise ValueError('input style '+str(inp)+' not recognized.')

    # read the command line arguments
    geom_file  = get_arg('-geom', geom_file)
    mo_file    = get_arg('-mos', mo_file)
    basis_file = get_arg('-basis', basis_file)
    gorder     = get_arg('-ordr', None)
    ci_file    = get_arg('-ci', None)
    out_file   = get_arg('-output', 'mos.dat')

    return input_style, geom_file, gorder, basis_file, mo_file, ci_file, out_file


def convert(inp, gm, go, basis, mos, ci, out):
    """Convert orbital/basis information to GAMESS format."""
    # import the appropriate module
    qc_input = __import__(inp, fromlist=['a'])

    if ci is not None:
        qc_input.generate_csf_list(ci)

    # parse the output and convert to GAMESS format
    gam_basis, gam_mos = qc_input.parse(gm, go, basis, mos)

    # print the MOs file
    gam_basis.print_basis_set(out)
    gam_mos.print_orbitals(out)
    if gam_mos.occ is not None:
        gam_mos.print_occ(out)


if __name__ == '__main__':
    # parse command line arguments
    args = process_arguments()

    # run the program
    convert(*args)
