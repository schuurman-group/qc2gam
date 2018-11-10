# 
# A set of standard structures to store basis set
# of MO information
#

import os
import math
import numpy as np

a_symbols       = [' H','HE',
                   'LI','BE','B ','C ','N ','O ','F ','NE',
                   'NA','MG','AL','SI','P ','S ','CL','AR']

a_numbers       = ['1.0','2.0',
                   '3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0',
                   '11.0','12.0','13.0','14.0','15.0','16.0','17.0','18.0']

ang_mom_sym     = ['S','P','D','F','G','H','I']

nfunc_per_shell = [1, 3, 6, 10, 15, 21, 28] 

# wrapper for all requisite data about an atom
class atom:

    def __init__(self, label, coords=None):
        # atomic symbol
        self.symbol    = label

        # set atomic number based on symbol
        try:
            self.number = a_numbers[a_symbols.index(self.symbol)]
        except:
            print('Atom: '+str(label)+' not found, 
                   setting atomic number to \'0\'\n')
            self.number = '0.0'

        # coordinates
        if coords is None:
            self.coords = np.zeros(3,dtype=float)
        else:
            self.coords = coords

    # define the coordinates of the atom
    def set_coords(self, coords):
        self.coords = coords

    # print the atom in GAMESS file format
    def print_atom(self, file_handle):
   
        ofmt   = ('{:2s}'+'{:>13s}'+
                  ''.join('{:18.10f}' for i in range(3))
        w_data = [self.symbol,self.number].extend(self.coords)

        file_handle.write(ofmt.format(*w_data))
            

# class to hold geometry information
class geom:

    def __init__(self, atoms=None):
        # if atoms is None, initialize empty atom list
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms

    def add_atom(self, atom):
        self.atoms.extend(atom)
        return

    def natoms(self):
        return len(self.atoms)   

# wrapper to hold information about the orbitals
class orbitals:

    def __init__(self, n_aos, n_mos):
        # number of AOs
        self.naos       = n_aos
        # number of MOs
        self.nmos       = n_mos
        # matrix holding MOs
        self.mo_vectors = np.zeros((n_aos,n_mos),dtype=float)

    # swap two orbitals
    def swap(self, mo_i, mo_j):
        if max(mo_i,mo_j) <= self.mo_vectors.shape[1]:
            mo_tmp = self.mo_vectors[:,mo_i]
            self.mo_vectors[:,mo_i] = self.mo_vectors[:,mo_j]
            self.mo_vectors[:,mo_j] = mo_tmp 
        return

    # add an orbital to the end of the list (i.e. at first column with norm==0
    def add(self, mo_vec):

        # only add orbital if the number of aos is correct
        if len(mo_vec) != self.n_aos:
            print("Cannot add orbital, wrong number of naos: "+
                   str(len(mo_vec))+"!="+str(self.naos))
            return

        i = 0
        while(np.linalg.norm(self.mo_vectors(:,i)) > fpzero):
            i += 1
        self.mo_vectors(:,i) = mo_vec
        return

    # insert an orbital
    def insert(self, mo_vec, mo_i)):
        self.mo_vectors = np.insert(self.mo_vectors,mo_i,mo_vec,axis=1)
        return

    # delete an orbital
    def delete(self, mo_i):
        self.mo_vectors = np.delete(self.mo_vectors,mo_i,axis=1)
        return

    # scale entire MO by a scalar
    def scale(self, mo_i, fac):
        self.mo_vectors(:,mo_i) *= fac
        return

    # scale each element of MO vector by a specific value
    def scale_vec(self, mo_i, fac_vec):
        self.mo_vectors(:,mo_i) *= fac_vec
        return

    # take the norm of an orbital
    def norm(self, mo_i):
        np.linalg.norm(self.mo_vectors(:,mo_i))
        return

    # convert mos from cartesian basis to spherical basis
    def cart2sph(self, basis_set):
    
        return

    # urconvert mos from spherical basis to cartesian basis
    def sph2cart(self, basis_set):

        return

    # print orbitals in gamess style VEC format
    def print_orbitals(self, file_name, n_orb=None):

        # default is to print all the orbitals
        if n_orb is None:
            n_orb = self.nmo

        # open file_name, append if file already exists
        with open(file_name, 'a') as mo_file:
            mo_file.write('$VEC\n')
            for i in range(n_orb):
                self.print_movec(mo_file, i)
            mo_file.write('$END\n')

        return

    # print an orbital vector
    def print_movec(self, file_handle, mo_i):
        n_col = 5 # this is set by GAMESS format
        n_row = int(math.ceil(self.naos/n_col))

        mo_row = ('{:>2s}'+' '+'{:>2s}'+''.join('{:15.8e}' for i in range(n_col))+'\n')
        mo_lab = mo_i % 100

        for i in range(n_row):
            r_data = [mo_lab,i].append(self.mo_vectors[5*i:5*(i+1),mo_i])
            file_handle.write(mo_row.format(*r_data)

        return

# an object to hold information about a single basis function
# including angular momentum, exponents and contraction coefficients
class basis_function:

    def __init_(self, ang_mom):
        self.ang_mom = ang_mom 
        self.ang_sym = ang_mom_sym[self.ang_mom]
        self.n_prim  = 0
        self.exps    = []
        self.coefs   = []

    def add_primitive(self, expo, coef):
        self.exps.extend(expo)
        self.coefs.extend(coef)
        self.n_prim += 1

    def print_basis_function(self, file_handle):
       
        ofmt1 = ('{:1s}'+'{:>6i}'+'\n')
        ofmt2 = ('{:>3i}'+'{:>15.7f}'+'{:>23.7f}'+'\n')

        w_data = [self.ang_sym, self.n_prim]
        file_handle.write(ofmt1.format(*w_data))
        for i in range(self.n_prim):
            w_data = [i, self.exps[i],self.coefs[i]]
            file_handle.write(ofmt1.format(*w_data))
        return

# contains the basis set information for a single atom
class basis_set:

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

    # add a basis function to the basis_set
    def add_function(self, atom_i, bf):
        self.basis_funcs[atom_i].extend(bf)
        self.n_cont[atom_i]      += 1
        self.basis_funcs[atom_i] += n_per_shell[bf.ang_mom]

    # print  the data section of a GAMESS style input file
    def print_basis_set(self, file_name):
    
        # if file doesn't exist, create it. Overwrite if it does exist
        with open(file_name, 'x') as dat_file:
            dat_file.write('$DATA\n'+
                           'Comment line | basis='+str(self.label)+'\n')

            for i in range(geom.natoms):
                geom.atoms[i].print_atom(dat_file)
                for j in range(self.n_cont[i])
                    self.basis_funcs[i][j].print_basis_function(dat_file)       
                # each atomic record ends with an empty line
                dat_file.write('\n')

            mo_file.write('$END\n')

        return

