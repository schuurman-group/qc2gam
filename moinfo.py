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

    # swap two orbitals
    def swap(self, mo_i, mo_j):
        """Documentation to come"""

        if max(mo_i,mo_j) <= self.mo_vectors.shape[1]:
            mo_tmp = self.mo_vectors[:,mo_i]
            self.mo_vectors[:,mo_i] = self.mo_vectors[:,mo_j]
            self.mo_vectors[:,mo_j] = mo_tmp 
        return

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

    # scale entire MO by a scalar
    def scale_mo(self, mo_i, fac):
        """Documentation to come"""
        self.mo_vectors[:,mo_i] *= fac
        return

    # scale each element of MO vector by a specific value
    def scale_mo_vec(self, mo_i, fac_vec):
        """Documentation to come"""
        self.mo_vectors[:,mo_i] *= fac_vec
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
            mo_file.write('$VEC\n')
            for i in range(n_orb):
                self.print_movec(mo_file, i)
            mo_file.write('$END\n')

        return

    # print an orbital vector
    def print_movec(self, file_handle, mo_i):
        """Documentation to come"""
        n_col = 5 # this is set by GAMESS format
        n_row = int(math.ceil(self.naos/n_col))

        mo_row = ('{:>2d}'+' '+'{:>2d}'+''.join('{:15.8E}' for i in range(n_col))+'\n')
        mo_lab = (mo_i+1) % 100

        for i in range(n_row):
            r_data = [mo_lab, i+1]
            r_data.extend(self.mo_vectors[5*i:5*(i+1),mo_i].tolist())
            file_handle.write(mo_row.format(*r_data))

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
        # total number of functions
        self.n_func      = 0
        # total number of contractions for atom i
        self.n_cont      = [0 for i in range(self.geom.natoms())]       
        # list of basis function objects
        self.basis_funcs = [[] for i in range(self.geom.natoms())]

    # add a basis function to the basis_set -- always keeps "like" angular 
    # momentum functions together
    def add_function(self, atom_i, bf):
        ang_mom = bf.ang_mom
        if len(self.basis_funcs[atom_i])>0:
            bf_i = 0
            while(ang_mom <= self.basis_funcs[atom_i][bf_i].ang_mom):
                bf_i += 1
                if bf_i == len(self.basis_funcs[atom_i]):
                    break
        else:
            bf_i = len(self.basis_funcs[atom_i]):

        self.basis_funcs[atom_i].insert(bf_i, bf)
        self.n_cont[atom_i] += 1
        self.n_func         += nfunc_cart[bf.ang_mom]
        return

    # print  the data section of a GAMESS style input file
    def print_basis_set(self, file_name):
    
        # if file doesn't exist, create it. Overwrite if it does exist
        with open(file_name, 'x') as dat_file:
            dat_file.write('$DATA\n'+
                           'Comment line | basis='+str(self.label)+'\n')

            for i in range(self.geom.natoms()):
                self.geom.atoms[i].print_atom(dat_file)
                for j in range(self.n_cont[i]):
                    self.basis_funcs[i][j].print_basis_function(dat_file)       
                # each atomic record ends with an empty line
                dat_file.write('\n')

            dat_file.write('$END\n')

        return

