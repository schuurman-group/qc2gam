# 
# A set of standard structures to store basis set
# of MO information
#

import os
import math
import numpy as np

a_symbols       = [' H','HE',
                   'LI','BE',' B',' C',' N',' O',' F','NE',
                   'NA','MG','AL','SI',' P',' S','CL','AR']

a_numbers       = [ '1.0', '2.0',
                    '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0','10.0',
                   '11.0','12.0','13.0','14.0','15.0','16.0','17.0','18.0']

ang_mom_sym     = ['S','P','D','F','G','H','I']

nfunc_cart      = [1, 3, 6, 10] 

nfunc_sph       = [1, 3, 5, 7]

ao_ordr         = [['s'],
                   ['px','py','pz'],
                   ['dxx','dyy','dzz','dxy','dxz','dyz'],
                   ['fxxx','fyyy','fzzz','fxxy','fxxz',
                    'fxyy','fyyz','fxzz','fyzz','fxyz']]

ao_norm         = [[1.],
                   [1.,1.,1.],
                   [1.,1.,1.,np.sqrt(3.),np.sqrt(3.),np.sqrt(3.)],
                   [1.,1.,1.,np.sqrt(5.),np.sqrt(5.),np.sqrt(5.),
                             np.sqrt(5.),np.sqrt(5.),np.sqrt(5.),np.sqrt(15.)]]

au2ang          = 0.529177249


# converts orbitals from spherical to cartesian
# AO basis
def sph2cart(basis, mo_sph, s2c):
    """Converts orbitals from spherical to cartesian AO basis. Assumes
       input orbitals are numpy array"""

    (nao_sph, nmo) = mo_sph.shape
    [l_cart, l_sph] = basis.ang_mom_lst()

    orb_trans = [[] for i in range(nmo)]
    iao_sph   = 0
    while(iao_sph < nao_sph):
        lval = l_sph[iao_sph]
        for imo in range(nmo):
            cart_orb = [sum([mo_sph[iao_sph+
                        s2c[lval][j][0][k],imo]*s2c[lval][j][1][k]
                        for k in range(len(s2c[lval][j][0]))])
                        for j in range(nfunc_cart[lval])]
            orb_trans[imo].extend(cart_orb)
        iao_sph  += nfunc_sph[lval]

    if iao_sph != basis.n_bf_sph:
        sys.exit('Error in sph2cart: '+str(iao_sph)+'!='+str(basis.n_bf_sph))

    mo_cart      = np.array(orb_trans).T
    return mo_cart

# wrapper for all requisite data about an atom
class atom:
    """Documentation to come"""

    def __init__(self, label, coords=None):
        """Documentation to come"""
        # atomic symbol
        self.symbol    = label

        # set atomic number based on symbol
        try:
            self.number = a_numbers[a_symbols.index(self.symbol)]
        except:
            print('Atom: '+str(label)+' not found,'+ 
                  'setting atomic number to \'0\'\n')
            self.number = '0.0'

        # coordinates
        if coords is None:
            self.coords = [0. for i in range(3)]
        else:
            self.coords = coords

    # define the coordinates of the atom
    def set_coords(self, coords):
        """Documentation to come"""
        self.coords = coords

    # print the atom in GAMESS file format
    def print_atom(self, file_handle):
        """Documentation to come"""
        ofmt   = ('{:2s}'+'{:>5s}'+
                  ''.join('{:18.10f}' for i in range(3))+'\n')
        w_data = [self.symbol,self.number] + self.coords
        file_handle.write(ofmt.format(*w_data))
            

# class to hold geometry information
class geom:
    """Documentation to come"""

    def __init__(self, atoms=None):
        """Documentation to come"""
        # if atoms is None, initialize empty atom list
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms

    def add_atom(self, atom):
        """Documentation to come"""
        self.atoms.append(atom)
        return

    def natoms(self):
        """Documentation to come"""
        return len(self.atoms)   

    def print_geom(self, file_handle):
        """Prints geometry in GAMESS format"""
        for i in range(self.natoms()):
            self.atoms[i].print_atom(file_handle)
  
# wrapper to hold information about the orbitals
class orbitals:
    """Documentation to come"""

    def __init__(self, n_aos, n_mos):
        """Documentation to come"""
        # number of AOs
        self.naos       = n_aos
        # number of MOs
        self.nmos       = n_mos
        # matrix holding MOs
        self.mo_vectors = np.zeros((n_aos,n_mos),dtype=float)

    # add an orbital to the end of the list (i.e. at first column with norm==0
    def add(self, mo_vec):
        """Documentation to come"""

        # only add orbital if the number of aos is correct
        if len(mo_vec) != self.naos:
            print("Cannot add orbital, wrong number of naos: "+
                   str(len(mo_vec))+"!="+str(self.naos))
            return

        i = 0
        while(np.linalg.norm(self.mo_vectors[:,i]) > 1.e-10):
            i += 1
        self.mo_vectors[:,i] = mo_vec
        return

   # insert an orbital
    def insert(self, mo_vec, mo_i):
        """Documentation to come"""
        self.mo_vectors = np.insert(self.mo_vectors,mo_i,mo_vec,axis=1)
        return

    # delete an orbital
    def delete(self, mo_i):
        """Documentation to come"""
        self.mo_vectors = np.delete(self.mo_vectors,mo_i,axis=1)
        return

    # scale each MO by the vector fac_vec
    def scale(self, fac_vec):
        """scale each MO by the vector fac_vec"""
        scale_fac       = np.array(fac_vec)
        old_mos         = self.mo_vectors.T
        new_mos         = np.array([mo * scale_fac for mo in old_mos]).T
        self.mo_vectors = new_mos

    # re-sort mos according to a map array
    def sort(self, map_lst):
        """re-sort the MOs, ordering the AO indices via map_lst"""
        for i in range(self.nmos):
            vec_srt = self.mo_vectors[map_lst,i]
            self.mo_vectors[:,i] = vec_srt 
        return

    # take the norm of an orbital
    def norm(self, mo_i):
        """Documentation to come"""
        np.linalg.norm(self.mo_vectors[:,mo_i])
        return

    # print orbitals in gamess style VEC format
    def print_orbitals(self, file_name, n_orb=None):
        """Documentation to come"""
        # default is to print all the orbitals
        if n_orb is None:
            n_orb = self.nmos

        # open file_name, append if file already exists
        with open(file_name, 'a') as mo_file:
            mo_file.write(' $VEC\n')
            for i in range(n_orb):
                self.print_movec(mo_file, i)
            mo_file.write(' $END\n')

        return

    # print an orbital vector
    def print_movec(self, file_handle, mo_i):
        """Documentation to come"""
        n_col  = 5 # this is set by GAMESS format
        n_row  = int(math.ceil(self.naos/n_col))
        mo_lab = (mo_i+1) % 100
 
        def mo_row(n):
            return ('{:>2d}'+' '+'{:>2d}'+''.join('{:15.8E}' for i in range(n))+'\n')

        for i in range(n_row):
            r_data = [mo_lab, i+1]
            n_coef  = min(self.naos,5*(i+1)) - 5*i
            r_data.extend(self.mo_vectors[5*i:min(self.naos,5*(i+1)),mo_i].tolist())
            file_handle.write(mo_row(n_coef).format(*r_data))
        return

# an object to hold information about a single basis function
# including angular momentum, exponents and contraction coefficients
class basis_function:
    """Documentation to come"""

    def __init__(self, ang_mom):
        # angular momentum of basis function
        self.ang_mom = ang_mom 
        # text symbol of the angular momentum shell
        self.ang_sym = ang_mom_sym[self.ang_mom]
        # number of primitives in basis
        self.n_prim  = 0
        # list of exponents
        self.exps    = []
        # list of coefficients
        self.coefs   = []

    def add_primitive(self, expo, coef):
        """Documentation to come"""
        self.exps.extend([expo])
        self.coefs.extend([coef])
        self.n_prim += 1
        return

    def print_basis_function(self, file_handle):
        """Documentation to come"""
        ofmt1 = ('{:1s}'+'{:>6d}'+'\n')
        ofmt2 = ('{:>3d}'+'{:>15.7f}'+'{:>23.7f}'+'\n')

        w_data = [self.ang_sym, self.n_prim]
        file_handle.write(ofmt1.format(*w_data))
        for i in range(self.n_prim):
            w_data = [i+1, self.exps[i],self.coefs[i]]
            file_handle.write(ofmt2.format(*w_data))
        return

# contains the basis set information for a single atom
class basis_set:
    """Documentation to come"""
    def __init__(self, label, geom):
        # string label of basis set name
        self.label       = label
        # nuclear coordinates (and coordinates of centers of gaussians
        self.geom        = geom
        # total number of cartesian functions
        self.n_bf_cart   = 0
        # total number of spherical functions
        self.n_bf_sph    = 0
        # total number of contractions for atom i
        self.n_bf        = [0 for i in range(self.geom.natoms())]       
        # list of basis function objects
        self.basis_funcs = [[] for i in range(self.geom.natoms())]

    # construct an array of the ang_mom of each basis function in 
    # GAMESS ordering -- returns both cartesian array and spherically
    # adapted array
    def ang_mom_lst(self):
        """Returns an array containing the value of angular momentum of the 
           corresponding at that index"""
        ang_mom_cart = []
        ang_mom_sph  = []

        for i in range(self.geom.natoms()):
            for j in range(self.n_bf[i]):
                ang_mom = self.basis_funcs[i][j].ang_mom
                ang_mom_cart.extend([ang_mom for k in range(nfunc_cart[ang_mom])])
                ang_mom_sph.extend([ang_mom for k in range(nfunc_sph[ang_mom])])

        return[ang_mom_cart, ang_mom_sph]

    # add a basis function to the basis_set -- always keeps "like" angular 
    # momentum functions together
    def add_function(self, atom_i, bf):
        ang_mom = bf.ang_mom
        if len(self.basis_funcs[atom_i])>0:
            bf_i = 0
            while(ang_mom >= self.basis_funcs[atom_i][bf_i].ang_mom):
                bf_i += 1
                if bf_i == len(self.basis_funcs[atom_i]):
                    break
        else:
            bf_i = len(self.basis_funcs[atom_i])

        self.basis_funcs[atom_i].insert(bf_i, bf)
        self.n_bf[atom_i]   += 1
        self.n_bf_cart      += nfunc_cart[bf.ang_mom]
        self.n_bf_sph       += nfunc_sph[bf.ang_mom]
        return

    # construct a mapping between the ordering of basis functions
    # in an arbitrary order and the 'canonical' GAMESS order
    # mapping also assumes 'canonical' cartesian representation
    # of AOs
    def construct_map(self, orig_ordr, orig_norm):
        """constructs a map array to convert the nascent AO ordering to
           GAMESS AO ordering. Also returns the normalization factors
           for the nascent and corresponding GAMESS-ordered AO basis"""
        map_arr       = []
        scale_nascent = []
        scale_gam     = []
        ao_map        = [[orig_ordr[i].index(ao_ordr[i][j]) 
                         for j in range(nfunc_cart[i])] 
                         for i in range(len(ao_ordr))]

        nf_cnt = 0
        for i in range(self.geom.natoms()):
            for j in range(len(self.basis_funcs[i])):
                ang_mom = self.basis_funcs[i][j].ang_mom
                nfunc   = nfunc_cart[ang_mom]
                map_bf = [nf_cnt + ao_map[ang_mom][k] for k in range(nfunc)]
                map_arr.extend(map_bf)
                scale_nascent.extend([orig_norm[ang_mom][k] for k in range(nfunc)])
                scale_gam.extend([1./ao_norm[ang_mom][k] for k in range(nfunc)])
                nf_cnt += nfunc

        return [map_arr, scale_nascent, scale_gam]


    # print  the data section of a GAMESS style input file
    def print_basis_set(self, file_name):
    
        # if file doesn't exist, create it. Overwrite if it does exist
        with open(file_name, 'x') as dat_file:
            dat_file.write(' $DATA\n'+
                           'Comment line | basis='+str(self.label)+'\n'+
                           'C1'+'\n')

            for i in range(self.geom.natoms()):
                self.geom.atoms[i].print_atom(dat_file)
                for j in range(self.n_bf[i]):
                    self.basis_funcs[i][j].print_basis_function(dat_file)       
                # each atomic record ends with an empty line
                dat_file.write('\n')

            dat_file.write(' $END\n')

        return

