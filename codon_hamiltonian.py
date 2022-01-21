import functools
from itertools import groupby
import operator
from collections import defaultdict

import numpy as np

#python codon table
import python_codon_tables as pct

#CAI library
from CAI import CAI
import Bio.Data.CodonTable as ct

#D-wave oceans
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite





class Amino_acid_to_Codon():
    def __init__(self, amino_acid_seq):
        self.amino_acid_seq = amino_acid_seq
        self.len_aa_seq = len(amino_acid_seq)

        # 
        self.list_all_possible_codons= self.to_possible_codons()
        
        self.N = len(_to_list(self.list_all_possible_codons)) 

    
    def __call__(self, selection=None):        
        if selection == None:
            return self.list_all_possible_codons
        else:
            #flattening
            flattening_all_possible_codons = sum(self.list_all_possible_codons, [])
            res_codon_frag = [flattening_all_possible_codons[i] for i in selection]
            return res_codon_frag

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
    


def fragmenting_amino_acid_seq(Amino_acid_seq, length_frag, ith):
    return Amino_acid_seq[length_frag * (ith):length_frag * (ith+1)]





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
        self.H_GC = [q_coefficients, qq_coefficients, const]

        "Minimizing sequentially repeated nucleotides term"
        self.H_R = wp['c_R'] * self.matrix_R()

        "Additional constraints"
        self.H_p = [(-1)*wp['epsilon']*np.ones(self.N), self.matrix_tau(wp['infty'])]

        "Conservation of the length of polypeptide"
        self.H_L = [wp['c_L'] * self.matrix_L()[0], wp['c_L'] * self.matrix_L()[1]]

        self.Q_ii = self.H_f + q_coefficients + self.H_p[0] + self.H_L[0]

        self.Q_ij = qq_coefficients + self.H_R + self.H_p[1] + self.H_L[1]

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


    def matrix_L(self):
        l = self.len_aa_seq
        diag_L = (1/(l**2) - 2/l) * np.ones(self.N)
        offdiag_L = np.ones((self.N,self.N))
        offdiag_L = 2/(l**2) * (np.triu(offdiag_L) - np.diag(np.diag(offdiag_L)))
        return [diag_L, offdiag_L]


    def _repeated_sequential_nucleotides(self, Ci, Cj):
        input = Ci + Cj
        groups = groupby(input)
        result = [(label, len(list(group))) for label, group in groups]
        list_counts = np.array(result)[:,1]
        outcome = np.max(list_counts.astype('int'))
        return outcome ** 2 - 1

    "BQM to Ising"
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

    "matrix_Q to dict_Q"
    def get_Q_dict(self):
        Q = dict()
        for i in range(len(self.Q_ii)):
            for j in range(i, len(self.Q_ii)):
                if i == j:
                    Q[(i,i)] = self.Q_ii[i]
                else:
                    Q[(i,j)] = self.Q_ij[i,j]
        return Q


    def run_Dwave(self, chain_strength=17, num_runs=10000):
        Q = self.get_Q_dict()

        sampler = EmbeddingComposite(DWaveSampler())
        self.response = sampler.sample_qubo(Q,
                                chain_strength=chain_strength,
                                num_reads=num_runs,
                                #num_spin_reversal_tramsforms=50,
                                label='Codon_Hamiltonian')

        return self._get_min_res()

    
    def _get_min_res(self):
        # Pick minimum Energy and its index
        min_E = min(list(self.response.data(fields=['energy'])))[0]
        min_index = list(self.response.data(fields=['energy'])).index(min_E)
        # Pick a sample associated with min Energy
        min_sample = list(self.response.data(fields=['sample']))[min_index][0]
        min_sample = [k for k,v in min_sample.items() if v == 1]
        
        return min_sample, min_E


    def outcome_codon_seq(self, base='RNA'):
        min_sample, _ = self._get_min_res()
        #flattening
        flattening_all_possible_codons = sum(self.list_all_possible_codons, [])

        if base == 'RNA':
            res_codon_frag = [flattening_all_possible_codons[i] for i in min_sample]

        elif base == 'DNA':
            res_codon_frag = [flattening_all_possible_codons[i].replace("U", "T") for i in min_sample]

        return res_codon_frag
        


class Run_whole_seq(Codon_Hamiltonian):
    def __init__(self, amino_acid_seq, wp, block_size=5, verbose=0):
        super().__init__(amino_acid_seq, wp)

        self.aminoacid_block = []
        self.dwave_opt_codons = []
        self.min_E_list = []

        for ith in range(len(amino_acid_seq) // block_size + 1):
            amino_fragment = fragmenting_amino_acid_seq(amino_acid_seq, block_size, ith)
            codon_fragment = Amino_acid_to_Codon(amino_fragment)

            
            if verbose >= 1:
                print('In amino acide seq, Run Block:',str(ith))
                print('=> Amino acids:', amino_fragment)
            if verbose >= 2:
                print('=> All possible codons:', codon_fragment())

            #run Dwave Sampler
            min_sample, min_E = super().run_Dwave() #chain_strength=15
            opt_codon_frag = super().outcome_codon_seq()
            
            if verbose >= 2:
                print('=> Ground states:', min_sample)
            if verbose >= 1:
                print('=> Optimal codons:', opt_codon_frag)

            self.aminoacid_block.append(amino_fragment)
            self.dwave_opt_codons.append(opt_codon_frag)
            self.min_E_list.append(min_E)


    def save_n_dp_outcome(self, DNA_name):
        return



def load_codon_seq(name, bs, wp):
    #bs: bloch_size

    #wp: weight params
    return






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


    def run_ExactDiag(self):
        #eigenval, eigenvec = eigsh(Model, k=1, which='SA')
        eigenval, eigenvec = np.linalg.eigh(self.hamiltonian)
        # real ground-state energy
        self.ground_energy= eigenval[0] + self.shift
        
        # Check the ground-state degeneracy
        degeneracy = np.where(np.isclose(eigenval[0], eigenval))[0][-1]

        if degeneracy == 0: 
            self.ground_state = eigenvec[:,0]
        else:
            self.groun_state = eigenvec[:,:degeneracy+1].transpose()


    def outcome_codon_seq(self, ):
        

        #flattening
        flattening_all_possible_codons = sum(self.list_all_possible_codons, [])

        if base == 'RNA':
            res_codon_frag = [flattening_all_possible_codons[i] for i in min_sample]

        elif base == 'DNA':
            res_codon_frag = [flattening_all_possible_codons[i].replace("U", "T") for i in min_sample]

        return res_codon_frag


    def get_outcome(self, types='bracket'):
        vec = self.ground_state
        
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
            res_binary = '0'*(num_qb-len(sigfig)) + sigfig 
            res['|'+res_binary+'>'] = vec[s]
        
        if types != 'bracket':
            res = [i for i,val in enumerate(res_binary) if val=='1']

        return res



def _to_list(list_of_list):
    return functools.reduce(operator.concat, list_of_list)







"""
Codon Adaption Index
"""


def getFlatPct(organ="ecoli"):
    if organ == "ecoli":
        table = pct.get_codons_table("e_coli_316407")
    elif organ == "human":
        table = pct.get_codons_table("h_sapiens_9606")

    tableForCAI={}
    for key in table.keys():
        for key2 in table[key].keys():
            tableForCAI[key2] = table[key][key2]
    return tableForCAI


def get_synonymous_codons(genetic_code_dict):

    # invert the genetic code dictionary to map each amino acid to its codons
    codons_for_amino_acid = {}
    for codon, amino_acid in genetic_code_dict.items():
        codons_for_amino_acid[amino_acid] = codons_for_amino_acid.get(amino_acid, [])
        codons_for_amino_acid[amino_acid].append(codon)

    # create dictionary of synonymous codons
    # Example: {'CTT': ['CTT', 'CTG', 'CTA', 'CTC', 'TTA', 'TTG'], 'ATG': ['ATG']...}
    return {
        codon: codons_for_amino_acid[genetic_code_dict[codon]]
        for codon in genetic_code_dict.keys()
    }


def get_RSCU_from_ctable(ctable, genetic_code=1): # genetic_code : ncbi_codon_table id ;  1 : standard
    _synonymous_codons = {
    k: get_synonymous_codons(v.forward_table) for k, v in ct.unambiguous_dna_by_id.items()
    }
    _non_synonymous_codons = {
        k: {codon for codon in v.keys() if len(v[codon]) == 1}
        for k, v in _synonymous_codons.items()
    }

    synonymous_codons = _synonymous_codons[genetic_code]
    counts = ctable
    result={}
    
    # calculate RSCU values
    for codon in ct.unambiguous_dna_by_id[genetic_code].forward_table:
        result[codon] = counts[codon] / (
            (len(synonymous_codons[codon]) ** -1)
            * (sum((counts[_codon] for _codon in synonymous_codons[codon])))
        )

    return result










class CAIs():
    def __init__(self, organ="ecoli"):
        self.organ = organ
        self.rscu_organ = self.get_RSCU_from_ctable()


    def __call__(self, codon_seq):
        codon_seq = self.check_basis(codon_seq)
        return CAI(codon_seq, RSCUs=self.rscu_organ)


    def getFlatPct(self):
        if self.organ == "ecoli":
            table = pct.get_codons_table("e_coli_316407")
        elif self.organ == "human":
            table = pct.get_codons_table("h_sapiens_9606")

        tableForCAI={}
        for key in table.keys():
            for key2 in table[key].keys():
                tableForCAI[key2] = table[key][key2]
        return tableForCAI


    def get_RSCU_from_ctable(self, genetic_code=1): # genetic_code : ncbi_codon_table id ;  1 : standard
        ctable = self.getFlatPct()

        _synonymous_codons = {
        k: self._get_synonymous_codons(v.forward_table) for k, v in ct.unambiguous_dna_by_id.items()
        }
        _non_synonymous_codons = {
            k: {codon for codon in v.keys() if len(v[codon]) == 1}
            for k, v in _synonymous_codons.items()
        }

        synonymous_codons = _synonymous_codons[genetic_code]
        counts = ctable
        result={}
        
        # calculate RSCU values
        for codon in ct.unambiguous_dna_by_id[genetic_code].forward_table:
            result[codon] = counts[codon] / (
                (len(synonymous_codons[codon]) ** -1)
                * (sum((counts[_codon] for _codon in synonymous_codons[codon])))
            )

        return result


    def _get_synonymous_codons(self, genetic_code_dict):

        # invert the genetic code dictionary to map each amino acid to its codons
        codons_for_amino_acid = {}
        for codon, amino_acid in genetic_code_dict.items():
            codons_for_amino_acid[amino_acid] = codons_for_amino_acid.get(amino_acid, [])
            codons_for_amino_acid[amino_acid].append(codon)

        # create dictionary of synonymous codons
        # Example: {'CTT': ['CTT', 'CTG', 'CTA', 'CTC', 'TTA', 'TTG'], 'ATG': ['ATG']...}
        return {
            codon: codons_for_amino_acid[genetic_code_dict[codon]]
            for codon in genetic_code_dict.keys()
        }




    
    def check_basis(self, codon_seq):
        try:
            codon_seq.index("T")
        except ValueError:
            codon_seq = codon_seq.replace("U", "T")

        return codon_seq

