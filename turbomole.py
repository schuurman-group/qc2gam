"""
A set of routines to parse the output of a turbomole
calculation.
"""
import sys
import numpy as np
import moinfo

# TURBOMOLE orbital ordering (in cartesians)
# s px py pz dxx dxy dxz dyy dyz dzz
# fxxx fxxy fxxz fxyy fxyz fxzz fyyy fyyz fyzz fzzz
ao_ordr = [['s'],
           ['px', 'py', 'pz'],
           ['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz'],
           ['fxxx', 'fyyy', 'fzzz', 'fxxy', 'fxxz',
            'fxyy', 'fyyz', 'fxzz', 'fyzz', 'fxyz']]

ao_norm = [[1.],
           [1., 1., 1.],
           [np.sqrt(3.), np.sqrt(3.), np.sqrt(3.),
            np.sqrt(3.), np.sqrt(3.), np.sqrt(3.)],
           [np.sqrt(15.), np.sqrt(15.), np.sqrt(15.), np.sqrt(15.),
            np.sqrt(15.), np.sqrt(15.), np.sqrt(15.), np.sqrt(15.),
            np.sqrt(15.), np.sqrt(15.)]]


# how to convert from spherical to cartesian basis functions
# (in turbomole ordering)
# s -> s | px -> px | py -> py | pz -> pz
# d ordering: d0, d1, d1-, d2-, d2
# dxx ->  0.5*( -d0/sqrt(3) + d2+ )
# dyy ->  0.5*( -d0/sqrt(3) - d2+ )
# dzz ->  d0 / sqrt(3)
# dxy ->  d2-
# dxz ->  d1+
# dyz ->  d1-

# f ordering: f0, f1, f1-, f2-, f2, f3, f3-
# fxxx -> -f1+/sqrt(40) + f3+/sqrt(24)
# fyyy -> -f1-/sqrt(40) + f3-/sqrt(24)
# fzzz ->  f0/sqrt(15)
# fxxy -> -f1-/sqrt(40) + f3- * sqrt(3/8)
# fxxz -> -f0 * sqrt(3/20) + f2 / 2
# fxyy -> -f1/sqrt(40) + f3 * sqrt(3/8)
# fyyz -> -f0 * sqrt(3/20) - f2 / 2
# fxzz ->  f1 * sqrt(2/5)
# fyzz ->  f1-* sqrt(2/5)
# fxyz ->  f2-
a        = 1./2.
b        = np.sqrt(2.)
c        = np.sqrt(3.)
d        = np.sqrt(5.)
f        = np.sqrt(6.)
g        = np.sqrt(10.)
h        = np.sqrt(15.)
sph2cart = [
    [[[0], [1.]]],                            # conversion for s orbitals
    [[[0], [1.]], [[1], [1.]], [[2], [1.]]],  # conversion for p orbitals
    [[[0, 4], [-a/c,  a]],                    # conversion for d orbitals
     [[0, 4], [-a/c, -a]],
     [[0], [1./c]],
     [[3], [1.]],
     [[1], [1.]],
     [[2], [1.]]],
    [[[1, 5], [-a/g, a/f]], # conversion for f orbitals
     [[2, 6], [-a/g, a/f]],
     [[0], [1./h]],
     [[2, 6], [-a/g, -a*c/b]],
     [[0, 4], [-a*c/d, a]],
     [[1, 5], [-a/g, -a*c/b]],
     [[0, 4], [-a*c/d, -a]],
     [[1], [b/d]],
     [[2], [b/d]],
     [[3], [1.]]]
            ]

def parse(geom_file, geom_ordr, basis_file, mo_file):
    """Parses a set of turbomole input files."""
    # read turbomole coord file
    gam_geom = read_geom(geom_file)
    if geom_ordr is not None:
        gam_ordr = read_geom(geom_ordr)
        gam_geom.reorder(gam_ordr)

    # parse basis file to get basis set information
    in_cart, gam_basis = read_basis(basis_file, gam_geom)

    # parse mos file to pull out orbitals
    gam_mos = read_mos(mo_file, in_cart, gam_basis)

    return gam_basis, gam_mos


def read_geom(geom_file):
    """Reads a turbomole coord file and loads results into moinfo.Geom."""
    # create geometry object to hold the data
    geom = moinfo.Geom()

    # slurp up geometry file
    with open(geom_file, 'r') as turbo_coord:
        gfile = turbo_coord.readlines()

    # parse geometry
    i = 0
    while gfile[i].split()[0] != '$coord':
        i += 1

    i += 1
    while '$' not in gfile[i]:
        gstr   = gfile[i].split() # format = coords, asym
        asym   = gstr[3].upper().rjust(2)
        coords = [float(gstr[j])*moinfo.au2ang for j in range(3)]
        geom.add_atom(moinfo.Atom(asym,coords))
        i     += 1

    return geom


def read_basis(basis_file, geom):
    """Reads a turbomole basis file into moinfo.BasisSet."""
    # create basis set object
    basis = moinfo.BasisSet('unknown', geom)

    # slurp up turbomole basis file
    with open(basis_file, 'r') as turbo_basis:
        bfile = turbo_basis.readlines()

    # default is to run in spherically adapted functions
    # can add for cartesians later (not sure how to check
    # that at the moment)
    in_cartesians = False

    # look for start of basis section
    i = 0
    while bfile[i].split()[0] != '$basis':
        i += 1

    # iterate over the atom types
    sec_start = False
    i        += 1
    while True:
        line = bfile[i].split()

        if line[0] == '$end':
            # stepping on $end line breaks us out of parser
            break
        elif len(line) == 0:
            # ignore blank lines
            i += 1
        elif line[0][0] == '#':
            # ignore comment lines
            i += 1
        elif line[0] == '*':
            # if first string is star, either
            # beginning or ending basis section
            sec_start = not sec_start
            i += 1
        elif sec_start:
            # if starting section, first line
            # is atom and basis set line
            a_sym = line[0]
            b_set = line[1]
            a_lst = [ind for ind,sym in enumerate([geom.atoms[k].symbol
                       for k in range(geom.natoms())])
                       if sym == a_sym.upper().rjust(2)]
            i += 1
        else:
            # if we get this far, we're parsing the basis set!
            nprim, ang_sym = line
            ang_mom        = moinfo.ang_mom_sym.index(ang_sym.upper())
            bfunc          = moinfo.BasisFunction(ang_mom)
            for j in range(int(nprim)):
                i += 1
                exp, coef = bfile[i].split()
                bfunc.add_primitive(float(exp), float(coef))
            for atom in a_lst:
                basis.add_function(atom, bfunc)

            i += 1

    return in_cartesians, basis


def read_mos(mocoef_file, in_cart, basis):
    """Reads a turbomole MO file into moinfo into moinfo.Orbitals.

    So: in the future, we should also allow for a mapping array that
    allows for taking linear combinations of elements in order to convert
    from a spherically adapted basis to cartesians. That will come later
    though... first the easy stuff.
    """
    # slurp up the mocoef file
    with open(mocoef_file, 'r') as mos:
        mo_file = mos.readlines()

    wid      = 20
    n_mos    = 0
    n_ao_all = []
    raw_orbs = []
    has_orb  = False
    for raw_line in mo_file:
        if has_orb:
            # ignore comment line, break at new section or end
            if raw_line[0] == '#':
                continue
            elif raw_line[0] == '$':
                break

            line = raw_line.split()
            if 'eigenvalue' in raw_line.lower():
                # if line contains "eigenvalue", start new orbitals
                raw_orbs.append([])
                n_ao_all.extend([0])
                n_mos += 1
            else:
                # else, parse a line
                mo_row = raw_line.lower().replace('d', 'e')
                n_cf = int(len(mo_row) // wid)
                for j in range(n_cf):
                    n_ao_all[-1] += 1
                    raw_orbs[-1].append(float(mo_row[j*wid:(j+1)*wid]))
        else:
            if '$scfmo' in raw_line or '$natural orbitals' in raw_line:
                has_orb = True

    # make sure all the mos are the same length
    uniq = list(set(n_ao_all))
    if len(uniq) != 1:
        raise ValueError('MOs have different numbers of AOs. Exiting...')
    else:
        n_aos = uniq[0]

    # create a numpy array to hold populations, if present
    turb_occ = np.zeros(n_mos, dtype=float)
    has_occ  = False
    for raw_line in mo_file:
        if has_occ:
            if raw_line[0] == '$':
                break
            else:
                line = raw_line.split()
                inds = np.array(line[1].split('-'), dtype=int)
                if len(inds) == 1:
                    turb_occ[inds[0]-1] = float(line[3])
                else:
                    turb_occ[inds[0]-1:inds[1]] = float(line[3])
        else:
            if 'occupation' in raw_line:
                has_occ = True

    if not has_occ:
        turb_occ = None

    # determine if we're running in cartesian or spherically adapted
    # atomic basis functions
    if n_aos == basis.n_bf_cart:
        in_cart = True
    elif n_aos == basis.n_bf_sph:
        in_cart = False
    else:
        raise ValueError('Number of basis functions is not equal to: ' +
                         '{:d}'.format(uniq[0]))

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
    gam_orb = moinfo.Orbitals(nao_cart, n_mos)
    gam_orb.mo_vectors = turb_orb_cart
    gam_orb.occ = turb_occ

    # construct mapping array from TURBOMOLE to GAMESS
    turb_gam_map, scale_turb, scale_gam = basis.construct_map(ao_ordr, ao_norm)

    # remove the turbomole normalization factors
    gam_orb.scale(scale_turb)

    # re-sort orbitals to GAMESS ordering
    gam_orb.sort_aos(turb_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb
