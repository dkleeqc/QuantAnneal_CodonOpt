import functools
from itertools import groupby
import operator
import numpy as np

import python_codon_tables as pct

class Amino_acid_to_Codon():
    def __init__(self, amino_acid_seq):
        self.amino_acid_seq = amino_acid_seq
        self.len_aa_seq = len(amino_acid_seq)

        # 
        self.list_all_possible_codons= self.to_possible_codons()
        
        self.N = len(_to_list(self.list_all_possible_codons)) 

    
    def __call__(self):        
        return self.list_all_possible_codons
    

    def to_possible_codons(self, amino_acids=None):
        all_possible_codons = []

        if amino_acids == None:
            lists = self.amino_acid_seq
        else:
            lists = amino_acids

        for x in lists:
            if x == 'F':
                all_possible_codons.append(sorted(['UUU', 'UUC']))
            elif x == 'L':
                all_possible_codons.append(sorted(['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG']))
            elif x == 'S':
                all_possible_codons.append(sorted(['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC']))
            elif x == 'Y':
                all_possible_codons.append(sorted(['UAU', 'UAC']))
            elif x == '*':
                all_possible_codons.append(sorted(['UAA', 'UAG', 'UGA']))
            elif x == 'C':
                all_possible_codons.append(sorted(['UGU', 'UGC']))
            elif x == 'W':
                all_possible_codons.append(['UGG'])


            elif x == 'P':
                all_possible_codons.append(sorted(['CCU', 'CCC', 'CCA', 'CCG']))
            elif x == 'H':
                all_possible_codons.append(sorted(['CAU', 'CAC']))
            elif x == 'Q':
                all_possible_codons.append(sorted(['CAA', 'CAG']))
            elif x == 'R':
                all_possible_codons.append(sorted(['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG']))


            elif x == 'I':
                all_possible_codons.append(sorted(['AUU', 'AUC', 'AUA']))
            elif x == 'M':
                all_possible_codons.append(['AUG'])
            elif x == 'T':
                all_possible_codons.append(sorted(['ACU','ACC','ACA','ACG']))
            elif x == 'N':
                all_possible_codons.append(sorted(['AAU','AAC']))
            elif x == 'K':
                all_possible_codons.append(sorted(['AAA', 'AAG']))


            elif x == 'V':
                all_possible_codons.append(sorted(['GUU', 'GUC', 'GUA', 'GUG']))
            elif x == 'A':
                all_possible_codons.append(sorted(['GCU', 'GCC', 'GCA', 'GCG']))
            elif x == 'D':
                all_possible_codons.append(sorted(['GAU', 'GAC']))
            elif x == 'E':
                all_possible_codons.append(sorted(['GAA', 'GAG']))
            elif x == 'G':
                all_possible_codons.append(sorted(['GGU', 'GGC', 'GGA', 'GGG']))


        return all_possible_codons



    def in_dna_base(self):
        return [[c.replace('U', 'T') for c in aa] for aa in self.list_all_possible_codons] # c: codon, aa: amino acid
    



class Codon_Hamiltonian(Amino_acid_to_Codon):
    def __init__(self, amino_acid_seq, wp):
        Amino_acid_to_Codon.__init__(self, amino_acid_seq)
        # wp: weight_params c_f, c_GC, c_R

        """
        codon_freqs = []
        counts_GC = []
        for aa in range(self.len_aa_seq):
            for c in codon_seq_in_dna_base[aa]:
                codon_freq = 
                counts_GC = 
        """
        "codon_usage_frequency term"
        self.H_f = wp['c_f'] * (-1) * self.vec_zeta(epsilon_f=wp['epsilon_f'])

        "Optimizing GC concentration term"
        s_i = self.vec_s()
        sigma_ij, square_s_i = self.matrix_ss()

        qq_coefficients = (2*wp['c_GC'] / self.N**2) * sigma_ij
        q_coefficients = (wp['c_GC'] / self.N**2) * square_s_i - 2 * (wp['rho_T'] * wp['c_GC'] / self.N) * s_i
        const = wp['c_GC'] * (wp['rho_T']**2)
        self.H_GC = [qq_coefficients, q_coefficients, const]

        "Minimizing sequentially repeated nucleotides term"
        self.H_R = wp['c_R'] * self.matrix_R()

        "Additional constraints"
        self.H_p = [(-1)*wp['epsilon']*np.ones(self.N), self.matrix_tau(wp['infty'])]

        self.Q_ii = self.H_f + q_coefficients + self.H_p[0]

        self.Q_ij = qq_coefficients + self.H_R + self.H_p[1]

    "codon_usage_frequency"
    def vec_zeta(self, host='e_coli_316407', epsilon_f=0.0001):
        codon_table = pct.get_codons_table(host)
        codon_seq_in_dna_base = self.in_dna_base()

        self.codon_freq = [[codon_table[self.amino_acid_seq[aa]][c] + epsilon_f for c in codon_seq_in_dna_base[aa]] for aa in range(self.len_aa_seq)]

        flattening_codon_freq = np.array(_to_list(self.codon_freq))
        return np.log(flattening_codon_freq)


    "for GC_contents_term"
    # s_i: the number of C's and G's in codon i 
    def vec_s(self):
        self.list_all_possible_codons
        
        self.counts_GC = [[c.count('G') + c.count('C') for c in self.list_all_possible_codons[aa]] for aa in range(self.len_aa_seq)]
        vec_s = _to_list(self.counts_GC)

        return np.array(vec_s)


    def matrix_ss(self):
        matrix_ss = np.outer(self.vec_s(), self.vec_s())
        diagonal_part = np.diag(matrix_ss)
        upper_triangle_part = np.triu(matrix_ss) - np.diag(diagonal_part)
        return upper_triangle_part, diagonal_part


    # minimizing_sequentially_repeated_nucleotides_term
    def matrix_R(self):
        Rmatrix = np.zeros((self.N, self.N))

        codon_list = self.list_all_possible_codons
        position_i = 0
        position_j = len(codon_list[0])
        for x in range(self.len_aa_seq-1):
            for a in range(len(codon_list[x])): # 
                for b in range(len(codon_list[x+1])): # 
                    #print(codon_list[x][a], codon_list[x+1][b])
                    #print(position_i+a, position_j+b)
                    Rmatrix[position_i+a, position_j+b] = self._repeated_sequential_nucleotides(codon_list[x][a], codon_list[x+1][b])
            
            position_i += len(codon_list[x])
            position_j += len(codon_list[x+1])

        return Rmatrix


    # Additional constraints
    def matrix_tau(self, infty=50):
        tau = np.zeros((self.N, self.N))
        t = 0 
        for x in range(len(self.list_all_possible_codons)):
            l = len(self.list_all_possible_codons[x]) 
            tau[t:t+l,t:t+l] = infty
            t += l
        return np.triu(tau - np.diag(np.diag(tau)))


    def _repeated_sequential_nucleotides(self, Ci, Cj):
        input = Ci + Cj
        groups = groupby(input)
        result = [(label, len(list(group))) for label, group in groups]
        list_counts = np.array(result)[:,1]
        outcome = np.max(list_counts.astype('int'))
        return outcome ** 2 - 1


    def Q_to_Jh(self):
        offdiag = self.Q_ij
        diag = self.Q_ii

        J = np.zeros((offdiag.shape))
        h = np.zeros(diag.shape)
        shift = 0
        for i in range(len(h)):
            shift += diag[i] / 2
            h[i] -= diag[i] / 2

            for j in range(len(h)):
                shift += offdiag[i][j] / 4
                J[i][j] = offdiag[i][j] / 4
                h[i] -= offdiag[i][j] / 4
                h[j] -= offdiag[i][j] / 4

                #shift += np.diag(diag)[i][j] / 2
                #h[i] -= np.diag(diag)[i][j] / 2

        return J, h, shift





class Quantum_Ising():
    """
    Construct a variety of interactions constructed in Hamiltonians
    """
    def __init__(self, J, h, shift):
        self.J = J
        self.h = h
        self.shift = shift
        self.L = len(h) 
        self.Z = np.array([[1, 0], [0, -1]])

        self.hamiltonian = self.JZZ() + self.hZ()


    def JZZ(self):
        res = np.zeros((2**self.L, 2**self.L))
        for i in range(1, self.L+1):
            for j in range(i+1, self.L+1):
                res += self.J[i-1][j-1] * self._ZZ_ij(i, j)
        return res


    def hZ(self):
        return np.sum([self.h[i-1] * self._Z_i(i) for i in range(1, self.L+1)], axis=0)


    def _ZZ_ij(self, i, j): # 1 <= i, j <= L
        if i >= j:
            raise ValueError("The index 'i' must be less than 'j'..!")
        ZZ = np.kron(self.Z, np.kron(np.eye(2**(j-i-1)), self.Z))
        return np.kron(np.eye(2**(i-1)), np.kron(ZZ, np.eye(2**(self.L-j))))


    def _Z_i(self, i): # 1 <= i <= L
        return np.kron(np.eye(2**(i-1)), np.kron(self.Z, np.eye(2**(self.L-i))))


    def ExactDiag(self):
        #eigenval, eigenvec = eigsh(Model, k=1, which='SA')
        eigenval, eigenvec = np.linalg.eigh(self.hamiltonian)
        # real ground-state energy
        self.GE= eigenval[0] + self.shift
        
        # Check the ground-state degeneracy
        degeneracy = np.where(np.isclose(eigenval[0], eigenval))[0][-1]

        if degeneracy == 0: 
            return eigenvec[:,0]
        else:
            return eigenvec[:,:degeneracy+1].transpose()






def vec_to_braket(vec):
    
    """
    vector to braket

    input: state vector as an array
    output: printing quantum states in bra-ket notation
    """
    
    num_qb = int(np.log(len(vec))/np.log(2))
    
    index_nonzero = np.where(np.isclose(vec, 0) == False)[0]
    res = dict()
    for s in index_nonzero:
        sigfig =bin(s)[2:]
        res['|'+'0'*(num_qb-len(sigfig)) + sigfig +'>'] = vec[s]
    
    return res




def _to_list(list_of_list):
    return functools.reduce(operator.concat, list_of_list)


def _codon_table_to_list(host):
    codon_table = pct.get_codons_table(host)

