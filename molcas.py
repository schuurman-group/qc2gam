"""
A set of routines to parse Molden format orbitals
"""
import sys
import numpy as np
import moinfo
from molden import read_geom, read_basis

# Molden orbital ordering (in cartesians)
# s px py pz dxx dyy dzz dxy dxz dyz
# fxxx fyyy fzzz fxyy fxxy fxxz fxzz fyzz fyyz fxyz
ao_ordr = [['s'],
           ['px', 'py', 'pz'],
           ['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz'],
           ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy',
            'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']]

ao_norm = [[1.],
           [1.,1.,1.],
           [1.,1.,1.,np.sqrt(3.),np.sqrt(3.),np.sqrt(3.)],
           [1.,1.,1.,np.sqrt(5.),np.sqrt(5.),np.sqrt(5.),
                     np.sqrt(5.),np.sqrt(5.),np.sqrt(5.),np.sqrt(15.)]]

# how to convert from spherical to cartesian basis functions
# (in molcas ordering)
# s -> s | px -> px | py -> py | pz -> pz
# d ordering: d0, d1+, d1-, d2+, d2-
# dxx ->  -d0 / 2 + sqrt(3.) * d2+ / 2
# dyy -> -d0 / 2 - sqrt(3.) * d2+ / 2
# dzz ->  d0
# dxy ->  d2-
# dxz ->  d1+
# dyz ->  d1-
# f ordering: f0, f1+, f1-, f2+, f2-, f3+, f3-
# fxxx -> -f1+
# fyyy -> -f1-
# fzzz -> -f0
# fxyy -> -(3/(2*sqrt(5)))*f1+ + (sqrt(3)/2)*f3+
# fxxy -> -(3/(2*sqrt(5)))*f1- + (sqrt(3)/2)*f3-
# fxxz -> -(3/(2*sqrt(5)))*f0  + (sqrt(3)/2)*f2+
# fxzz -> -(3/(2*sqrt(5)))*f1+ - (sqrt(3)/2)*f3+
# fyzz -> -(3/(2*sqrt(5)))*f1- - (sqrt(3)/2)*f3-
# fyyz -> -(3/(2*sqrt(5)))*f0  - (sqrt(3)/2)*f2+
# fxyz -> f2-
sph2cart = [
    [[[0], [1.]]],                           # conversion for s orbitals
    [[[0], [1.]], [[1], [1.]], [[2], [1.]]], # conversion for p orbitals
    [[[0, 3], [-1./2., np.sqrt(3.)/2.]],     # conversion for d orbitals
     [[0, 3], [-1./2., -np.sqrt(3.)/2.]],
     [[0], [1.]],
     [[4], [1.]],
     [[1], [1.]],
     [[2], [1.]]],
    [[[1, 5], [1., 0.]],                     # conversion for f orbitals
     [[2, 6], [1., 0.]],
     [[0], [1.]],
     [[1, 5], [-3./(2.*np.sqrt(5.)), np.sqrt(3.)/2.]],
     [[2, 6], [-3./(2.*np.sqrt(5.)), np.sqrt(3.)/2.]],
     [[0, 3], [-3./(2.*np.sqrt(5.)), np.sqrt(3.)/2.]],
     [[1, 5], [-3./(2.*np.sqrt(5.)), -np.sqrt(3.)/2.]],
     [[2, 6], [-3./(2.*np.sqrt(5.)), -np.sqrt(3.)/2.]],
     [[0, 3], [-3./(2.*np.sqrt(5.)), -np.sqrt(3.)/2.]],
     [[4], [1.]]]
            ]


def parse(geom_file, geom_ordr, basis_file, mo_file):
    """Parses a set of molcas input files.

    Molcas orbital files do not contain geometry or basis. Instead,
    a molden file is parsed. Only one of basis_file and geom_file
    need to be specified.
    """
    # set unset values
    if geom_file is None:
        geom_file = basis_file
    elif basis_file is None:
        basis_file = geom_file

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


def read_mos(mocoef_file, in_cart, basis):
    """Reads a Molcas orbital file into moinfo into moinfo.Orbitals.

    So: in the future, we should also allow for a mapping array that
    allows for taking linear combinations of elements in order to convert
    from a spherically adapted basis to cartesians. That will come later
    though... first the easy stuff.
    """
    # slurp up the molcas file
    with open(mocoef_file, 'r') as mos:
        mo_file = mos.readlines()

    nline = len(mo_file)
    for il in range(nline):
        if '#INFO' in mo_file[il]:
            # lines after: 1. comment, 2. uhf, nsym, 0, 3. nbas, 4. nbas
            n_mos = int(mo_file[il+3])
            break

    has_orb = False
    for il in range(il+1, nline):
        if '#ORB' in mo_file[il]:
            has_orb = True
            break

    if not has_orb:
        raise IOError('MOs not found in input file.')

    imo = -1
    n_ao_all = []
    raw_orbs = []
    for il in range(il+1, nline):
        if 'ORBITAL' in mo_file[il]:
            imo += 1
            raw_orbs.append([])
            n_ao_all.extend([0])
        elif imo < 0:
            raise ValueError('#ORB section must start with ORBITAL')
        elif '#' in mo_file[il]:
            # new section, break
            break
        else:
            mo_row = mo_file[il].split()
            n_ao_all[-1] += len(mo_row)
            raw_orbs[-1] += [float(o) for o in mo_row]

    # make sure all mos are the same length
    uniq = list(set(n_ao_all))
    if len(uniq) != 1:
        raise ValueError('MOs have different numbers of AOs. Exiting...')
    else:
        n_aos = uniq[0]

    # create a numpy array to hold populations, if present
    imo = 0
    molc_occ = np.zeros(n_mos, dtype=float)
    has_occ = False
    for il in range(il, nline):
        if has_occ:
            if 'OCCUPATION' in mo_file[il]:
                continue
            elif '#' in mo_file[il]:
                # new section, break
                break
            else:
                occ_row = mo_file[il].split()
                molc_occ[imo:imo+len(occ_row)] = [float(o) for o in occ_row]
                imo += len(occ_row)
        else:
            if '#OCC' in mo_file[il]:
                has_occ = True

    if np.allclose(molc_occ, 0):
        molc_occ = None
    else:
        molc_occ = np.array(molc_occ)

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
    molc_orb = np.array(raw_orbs).T
    if not in_cart:
        molc_orb_cart = moinfo.sph2cart(basis, molc_orb, sph2cart)
        nao_cart      = molc_orb_cart.shape[0]
    else:
        molc_orb_cart = molc_orb
        nao_cart      = n_aos

    # copy orbitals into gam_orb
    gam_orb = moinfo.Orbitals(nao_cart, n_mos)
    gam_orb.mo_vectors = molc_orb_cart
    gam_orb.occ = molc_occ

    # construct mapping array from TURBOMOLE to GAMESS
    molc_gam_map, scale_molc, scale_gam = basis.construct_map(ao_ordr, ao_norm)

    # remove the molcas normalization factors
    gam_orb.scale(scale_molc)

    # re-sort orbitals to GAMESS ordering
    gam_orb.sort_aos(molc_gam_map)

    # apply the GAMESS normalization factors
    gam_orb.scale(scale_gam)

    return gam_orb
