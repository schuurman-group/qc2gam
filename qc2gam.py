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
        basis_file = 'basis'
        mo_file = 'mos'
    elif input_style == 'columbus':
        geom_file = 'geom'
        basis_file = 'daltaoin'
        mo_file = 'mocoef'
    elif input_style == 'molcas':
        geom_file = 'mos.molden'
        basis_file = None
        mo_file = 'INPORB'
    elif input_style == 'molden':
        geom_file = None
        basis_file = None
        mo_file = 'mos.molden'
    else:
        raise ValueError('input style '+str(input_style)+' not recognized.')

    # read the command line arguments
    geom_file  = get_arg('-geom', geom_file)
    basis_file = get_arg('-basis', basis_file)
    mo_file    = get_arg('-mos', mo_file)
    gorder     = get_arg('-ordr', None)
    ci_file    = get_arg('-ci', None)
    out_file   = get_arg('-output', 'mos.dat')

    return input_style, geom_file, gorder, basis_file, mo_file, ci_file, out_file


def convert(inp, gm, go, basis, mos, ci, out):
    """Convert orbital/basis information to GAMESS format."""
    # import the appropriate module
    qc_input = __import__(inp, fromlist=['a'])

    # if the determinant list only involves the first nmo_prnt, we only
    # should print the first nmo_prnt orbs to file
    if ci is not None:
        nmo_prnt = qc_input.generate_csf_list(ci)
    else:
        # default is to print all MOs on file
        nmo_prnt = -1
 
    # parse the output and convert to GAMESS format
    gam_basis, gam_mos = qc_input.parse(gm, go, basis, mos)

    # print the MOs file
    gam_basis.print_basis_set(out)
    gam_mos.print_orbitals(out, n_orb = nmo_prnt)
    if gam_mos.occ is not None:
        gam_mos.print_occ(out, n_orb = nmo_prnt)


if __name__ == '__main__':
    # parse command line arguments
    args = process_arguments()

    # run the program
    convert(*args)
