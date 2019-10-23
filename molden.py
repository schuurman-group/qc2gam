"""
A set of routines to parse Molden format orbitals
"""
import sys
import numpy as np
import moinfo

# Molden orbital ordering (in cartesians)
# s px py pz dxx dxy dxz dyy dyz dzz
# fxxx fxxy fxxz fxyy fxyz fxzz fyyy fyyz fyzz fzzz
ao_ordr = [['s'],
           ['px', 'py', 'pz'],
           ['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz'],
           ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy',
            'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']]

ao_norm = [[1.],
           [1.,1.,1.],
           [1.,1.,1.,np.sqrt(3.),np.sqrt(3.),np.sqrt(3.)],
           [np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
            np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),np.sqrt(15.),
            np.sqrt(15.),np.sqrt(15.)]]

# how to convert from spherical to cartesian basis functions
# (in molden ordering)
# s -> s | px -> px | py -> py | pz -> pz
# d ordering: d0, d1+, d1-, d2+, d2-
# dxx ->  -d0 / 6 + d2+ / 2
# dxy ->  d2-
# dxz ->  d1+
# dyy -> -d0 / 6 - d2+ / 2
# dyz ->  d1-
# dzz ->  d0 / 3
# f ordering: f0, f1+, f1-, f2+, f2-, f3+, f3-
# fxxx -> -3*(f1+ / 20 + f3+ / 4)
# fxxy -> f3- / 4 - f1- / 20
# fxxz -> 3*f0 / 10 + f2+ / 2
# fxyy -> 3*f3+ / 4 - f1+ / 20
# fxyz -> f2-
# fxzz -> f1+ / 5
# fyyy -> -3*f1- / 20 - f3- / 4
# fyyz -> 3*f0 / 10 - f2+ / 2
# fyzz -> f1- / 5
# fzzz -> -3*f0 / 5
sph2cart = [
    [[[0], [1.]]],                           # conversion for s orbitals
    [[[0], [1.]], [[1], [1.]], [[2], [1.]]], # conversion for p orbitals
    [[[0, 3], [-1./6., 1./2.]],              # conversion for d orbitals
     [[0, 3], [-1./6., -1./2.]],
     [[0], [1./3.]],
     [[4], [1.]],
     [[1], [1.]],
     [[2], [1.]]],
    [[[1, 5], [-3./20., -3./4.]],            # conversion for f orbitals
     [[2, 6], [-3./20., -1./4.]],
     [[0], [-3./5.]],
     [[1, 5], [-1./20., 3./4.]],
     [[2, 6], [-1./20., 1./4.]],
     [[0, 3], [3./10., 1./2.]],
     [[1], [1./5.]],
     [[2], [1./5.]],
     [[0, 3], [3./10., -1./2.]],
     [[4], [1.]]]
            ]


def parse(geom_file, geom_ordr, basis_file, mo_file):
    """Parses a set of molden input files.

    Molden files contain geometry, basis and MO information. If
    geom_file or basis_file are None, they are set to the same
    value as mo_file.
    """
    # set unset values
    if geom_file is None:
        geom_file = mo_file
    if basis_file is None:
        basis_file = mo_file

    # read xyz coord file
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
    """Reads a molden file and loads results into moinfo.Geom."""
    # create geometry object to hold the data
    geom = moinfo.Geom()

    # slurp up geometry file
    with open(geom_file, 'r') as molden_coord:
        gfile = molden_coord.readlines()

    nline = len(gfile)

    # parse geometry
    for i in range(nline):
        if '[N_ATOMS]' in gfile[i].upper():
            natm = int(gfile[i+1].split()[0])
            i += 2
            break

    for i in range(i, nline):
        if '[ATOMS]' in gfile[i].upper():
            if '(AU)' in gfile[i].upper():
                conv = moinfo.au2ang
            else:
                conv = 1.

            i += 1
            break

    nums = '0123456789'
    for k in range(natm):
        gstr   = gfile[i].split() # format = asym, num, anum, coords
        asym   = gstr[0].rstrip(nums).upper().rjust(2)
        coords = [float(gstr[j+3])*conv for j in range(3)]
        geom.add_atom(moinfo.Atom(asym, coords))
        i     += 1

    return geom


def read_basis(basis_file, geom):
    """Reads a molden file into moinfo.BasisSet."""
    # create basis set object
    basis = moinfo.BasisSet('unknown', geom)

    # slurp up molden file
    with open(basis_file, 'r') as molden_basis:
        bfile = molden_basis.readlines()

    nline = len(bfile)

    # default is to run in spherically adapted functions
    # can add for cartesians later (in Molden, this can be
    # done by searching for lines [6D], [10F]
    in_cartesians = False

    # look for start of basis section
    for i in range(nline):
        if '[GTO]' in bfile[i].upper():
            i += 1
            break

    # iterate over each atom
    iatm = 0
    it = iter(range(i, nline))
    for i in it:
        line = bfile[i].split()

        # stepping on new section line breaks us out of parser,
        # blank line separates atoms, single element check
        # atom index, otherwise read the basis functions
        if len(line) == 0:
            iatm += 1
        elif '[' in line[0]:
            break
        elif len(line) == 1:
            if int(line[0]) != iatm + 1:
                raise ValueError('Atomic index mismatch: '+str(iatm+1)+
                                 ', '+line[0])
        else:
            ang_sym, nprim = line
            ang_mom        = moinfo.ang_mom_sym.index(ang_sym.upper())
            bfunc          = moinfo.BasisFunction(ang_mom)
            for j in range(int(nprim)):
                i = next(it)
                exp, coef = bfile[i].split()
                bfunc.add_primitive(float(exp), float(coef))

            basis.add_function(iatm, bfunc)

    return in_cartesians, basis


def read_mos(mocoef_file, in_cart, basis):
    """Reads a molden file into moinfo into moinfo.Orbitals.

    So: in the future, we should also allow for a mapping array that
    allows for taking linear combinations of elements in order to convert
    from a spherically adapted basis to cartesians. That will come later
    though... first the easy stuff.
    """
    # slurp up the molden file
    with open(mocoef_file, 'r') as mos:
        mo_file = mos.readlines()

    n_mos    = 0
    n_ao_all = []
    raw_orbs = []
    mold_occ = []
    has_orb  = False
    for raw_line in mo_file:
        if has_orb:
            # ignore comment line, break at new section or end
            if raw_line[0] == '[':
                break

            line = raw_line.split()
            if 'sym' in raw_line.lower():
                # if line contains "sym", start new orbitals
                raw_orbs.append([])
                n_ao_all.extend([0])
                n_mos += 1
            elif 'ene' in raw_line.lower():
                continue
            elif 'spin' in raw_line.lower():
                continue
            elif 'occup' in raw_line.lower():
                mold_occ.append(float(line[1]))
            else:
                # else, parse a line
                n_ao_all[-1] += 1
                raw_orbs[-1].append(float(line[1]))
        else:
            if '[MO]' in raw_line.upper():
                has_orb = True

    # make sure all the mos are the same length
    uniq = list(set(n_ao_all))
    if len(uniq) != 1:
        raise ValueError('MOs have different numbers of AOs. Exiting...')
    else:
        n_aos = uniq[0]

    # create a numpy array to hold populations, if present
    if np.allclose(mold_occ, 0):
        mold_occ = None
    else:
        mold_occ = np.array(mold_occ)

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
    mold_orb = np.array(raw_orbs).T
    if not in_cart:
        mold_orb_cart = moinfo.sph2cart(basis, mold_orb, sph2cart)
        nao_cart      = mold_orb_cart.shape[0]
    else:
        mold_orb_cart = mold_orb
        nao_cart      = n_aos

    # copy orbitals into gam_orb
    gam_orb = moinfo.Orbitals(nao_cart, n_mos)
    gam_orb.mo_vectors = mold_orb_cart
    gam_orb.occ = mold_occ

    # construct mapping array from TURBOMOLE to GAMESS
    mold_gam_map, scale_mold, scale_gam = basis.construct_map(ao_ordr, ao_norm)

    # remove the dalton normalization factors
    gam_orb.scale(scale_mold)

    # re-sort orbitals to GAMESS ordering
    gam_orb.sort(mold_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb
