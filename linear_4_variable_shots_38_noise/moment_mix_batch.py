#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from itertools import product 
import shadows
import json
import qiskit as qm
import Hamiltonians.helper as helper
import Hamiltonians.hamiltonians as ham
import numpy as np
from itertools import combinations
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
import sys
import warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from tqdm import tqdm
from qiskit import IBMQ

n_qubits = 4# int(input())
n_moments_exact = 2#int(input())
K = 5
print("N Qubits: ", n_qubits)
print("Maximum K: ", K)
print("Moments Exact: ", n_moments_exact)
print("Enter Number of Rounds: ")
n_rounds = int(input())
print("Enter Save Directory: ")
save_dir = str(input())
print("----------------------------------------")
print("Running ", n_qubits, " for ", n_rounds, " with ", n_moments_exact, " QWC moments")
ket0 = np.array([1.,0.],dtype=complex)
ket1 = np.array([0.,1.],dtype=complex)


PauliX = np.array([[0.,1.],[1.,0]], dtype=complex)
PauliY = np.array([[0.,-1.j],[1.j,0]], dtype=complex)
PauliZ = np.array([[1.0,0.],[0.,-1.]], dtype=complex)
PauliI = np.eye(2, dtype=complex)
PauliVec = [PauliX, PauliY, PauliZ]
cz = np.eye(4)
cz[3,3] = -1
PauliDict = {"Identity": "I", "PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}
PauliNumToLetter = {0: "X", 1: "Y", 2: "Z"}
PauliDict_2 = {"I": PauliI, "X": PauliX, "Y": PauliY, "Z": PauliZ}
Hadamard = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]])
CNOT = np.eye(4)
CNOT[3,3]=0
CNOT[2,2] = 0 
CNOT[3,2] = 1
CNOT[2,3] = 1
thetaTest = [-2,1]
cz = np.eye(4)
cz[3,3] = -1


"""

FUNCTION DEFINITIONS


"""

def RY(theta):
    '''
        matrix representation of the RY gate in the Z computational basis
    '''
    matrix = np.zeros((2,2),dtype=complex)
    matrix[0,0] = np.cos(theta/2)
    matrix[1,1] = np.cos(theta/2)
    matrix[0,1] = -np.sin(theta/2)
    matrix[1,0] =  np.sin(theta/2)
    return matrix
def RZ(theta):
    matrix = np.zeros((2,2), dtype=complex)
    matrix[0,0] = np.exp(-1.j * theta/2)
    matrix[1,1] = np.exp(1.j * theta/2)
    return matrix
def linear_hamiltonian_matrix_dict(j,u,b=1.0,n=10):
    '''
        represent our Hamiltonian as a matrix
    '''
    combos = list(combinations([int(i) for i in range(n)],2))
    H = {}
    for i in range(n-1):
        next_ = i+1

        for k in range(3):
            A = ["I" for i in range(n)]
            A[i] = PauliNumToLetter[k]
            A[next_] = PauliNumToLetter[k]
            x = j
            if(k==2):
                x = u
            H["".join(A)] = x
            
    for i in range(n):
        A = ["I" for i in range(n)]
        A[i] = "Z"
        H ["".join(A)] = b
    return H

def linear_hamiltonian_matrix(j,u,b=1.0,n=10):
    '''
        represent our Hamiltonian as a matrix
    '''
    combos = list(combinations([int(i) for i in range(n)],2))
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n-1):
        next_ = i+1

        for k in range(3):
            A = [PauliI for i in range(n)]
            A[i] = PauliVec[k]
            A[next_] = PauliVec[k]
            x = j
            if(k==2):
                x = u
            add = [1]
            for qbit in range(n):
                
                add = np.kron(add, A[qbit])
            H += x*add
            
    for i in range(n):
        A = [PauliI for i in range(n)]
        A[i] = PauliZ
        add = [1]
        for qbit in range(n):
                add = np.kron(add, A[qbit])
        H += b * add
    return H


def circuit(theta):
    qc = qm.QuantumCircuit(n_qubits,n_qubits)
    for w in range(n_qubits-1):
        qc.h(w)
        qc.ry(theta[w], w)
    for w in range(n_qubits-1):
        qc.cnot(w, w+1)
    for w in range(n_qubits):
        qc.rz(theta[w + n_qubits], w)
    return qc


# In[6]:


def circuit_numerical(theta):
    psi = np.zeros(2 ** n_qubits)
    print(2**n_qubits)
    psi[0] = 1 # initialize in |0000....000>
    for w in range(n_qubits-1):
        obs_ = [1]
        for i in range(n_qubits):
            if not i == w:
                obs_ = np.kron(obs_, PauliI)
            else:
                obs_ = np.kron(obs_, Hadamard)
        psi = obs_ @ psi 
        obs_ = [1]
        for i in range(n_qubits):
            if not i == w:
                obs_ = np.kron(obs_, PauliI)
            else:
                obs_ = np.kron(obs_, RY(theta[w]))
        psi = obs_ @ psi 
    for w in range(n_qubits-1):
        obs_ = [1]
        for i in range(n_qubits):
            if not i == w and not i == w+1:
                obs_ = np.kron(obs_, PauliI)
            elif i == w:
                obs_ = np.kron(obs_, CNOT)
            elif i == w+1:
                pass;
            
        psi = obs_ @ psi
    for w in range(n_qubits):
        obs_ = [1]
        for i in range(n_qubits):
            if not i == w:
                obs_ = np.kron(obs_, PauliI)
            else:
                obs_ = np.kron(obs_, RZ(theta[w+n_qubits]))
        psi = obs_ @ psi
    return psi




#PDS(K) workhorse 

def tildeM(e_dict, K):
    M = np.zeros((K,K), dtype=complex)
    for i in range(1,K+1):
        for j in range(1,K+1):
            M[i-1,j-1] = e_dict[int(2*K-i-j)]
    return M

def tildeY(e_dict, K):
    Y = np.zeros(K, dtype=complex)
    for i in range(1,K+1):
        Y[i-1] = e_dict[int(2*K-i)]
    return Y
def comp_energy(moments, K):
    M_ = tildeM(moments, K)
    Y_ = tildeY(moments, K)
    X_ = np.linalg.lstsq(M_, -Y_,rcond=None)[0]
    coeffs = np.ones(K+1)
    for k in range(1,K+1):
        coeffs[k] = X_[k-1].real
    return np.min(np.roots(coeffs)).real


# In[10]:


# Using Dr. Peng's derivation

def dYdH(K,n):
    return np.array([1 * (n == 2*K-i) for i in range(1,K+1)])
def dMdH(K, n):
    return np.array([[1 * (n == 2*K-i-j) for i in range(1, K+1)] for j in range(1,K+1)])
def dXdH(K,n,M,X):
    dydh = dYdH(K,n)
    dmdh = dMdH(K,n)
    dXdH = np.linalg.lstsq(M, -dydh - dmdh@X, rcond=None)[0]
    return dXdH
def dEdH(E,K,n,M,Y,X):
    dxdh = dXdH(K, n, M, X)
    Evec = np.array([E ** (K-1-i) for i in range(K)])
    bottom_of_coeff = K*E**(K-1) + np.sum([(K-i)*X[i-1] * E**(K-i-1) for i in range(1,K)])
    return (-1/bottom_of_coeff) * (np.dot(Evec,dxdh))
def calculate_moment_partials(moments_dict_):
    M_est = tildeM(moments_dict_, K)
    Y_est = tildeY(moments_dict_, K)
    X_est = np.linalg.lstsq(M_est, -Y_est,rcond=None)[0]
    E = comp_energy(moments_dict_, K)
    partial_derivatives = [dEdH(E, K, i, M_est, Y_est, X_est) for i in range(1,2*K)]
    return partial_derivatives

def pauli_importances_from_moments(m_dict_):
    M_est_ = tildeM(m_dict_, K)
    Y_est_ = tildeY(m_dict_, K)
    X_est_ = np.linalg.lstsq(M_est_, -Y_est_,rcond=None)[0]
    E_ = comp_energy(m_dict_, K)
    m_p_ = [dEdH(E_, K, i, M_est_, Y_est_, X_est_) for i in range(1,2*K)]
    m_pauli_partials_ = {}
    ham_qm_powers = [ham_qm.power(i) for i in range(0,2*K)]
    for i in range(2*K-1): # this is poorly written and has confusing indeces
                           # but it does do the job
        letters = ham_qm_powers[i+1].letters
        coeffs = ham_qm_powers[i+1].coeffs
        for j in range(len(letters)):
            if(letters[j] in m_pauli_partials_):
                m_pauli_partials_[letters[j]] += (m_p_[i] * coeffs[j])
            else:
                m_pauli_partials_[letters[j]] = (m_p_[i] * coeffs[j])
    return m_pauli_partials_

def stringToBasisState(string):
    '''
        Convert a Pauli Word to an array of its eigenvalues
    
    '''
    ket0 = np.array([1.,0.])
    ket1 = np.array([0., 1.])
    ans = [1]
    for i in range(len(string)):
        if(string[i]=="1"):
            ans = np.kron(ans, ket1)
        else:
            ans = np.kron(ans, ket0)
    return ans

def expectationFromGroupMeasurment(measurement, obs):
    
    '''
        Convert a computational basis measurement to the expectation value of an observable
    
    ''' 
    pauliEigen = np.array([1.,-1.])
    identityEigen = np.array([1.,1.])
    eigen = [pauliEigen]*len(obs)
    for i in range(len(eigen)):
        if(obs[i]=="I"):
            eigen[i] = identityEigen
    e = [1]
    for i in range(len(obs)):
        e = np.kron(e, eigen[i])
    eigen = e
    return measurement @ eigen
   
def circuit(theta):
    qc = qm.QuantumCircuit(n_qubits,n_qubits)
    for w in range(n_qubits-1):
        qc.h(w)
        qc.ry(theta[w], w)
    for w in range(n_qubits-1):
        qc.cnot(w, w+1)
    for w in range(n_qubits):
        qc.rz(theta[w + n_qubits], w)
    return qc

from scipy.optimize import curve_fit

def func(x, popt):
    return np.exp(popt[1]) * np.exp(popt[0] * x)
def approximate_moment(i, popt):
    sgn = np.abs(exact_moments[3])/exact_moments[3]
    return (sgn)**i * func(i, popt)

def shadow_bound(error, observables, failure_rate=0.01):
    """
    Calculate the shadow bound for the Pauli measurement scheme.

    Implements Eq. (S13) from https://arxiv.org/pdf/2002.08953.pdf

    Args:
        error (float): The error on the estimator.
        observables (list) : List of matrices corresponding to the observables we intend to
            measure.
        failure_rate (float): Rate of failure for the bound to hold.

    Returns:
        An integer that gives the number of samples required to satisfy the shadow bound and
        the chunk size required attaining the specified failure rate.
    """
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - (np.eye(len(op))*np.trace(op) / 2 ** int(np.log2(op.shape[0]))), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error ** 2
    return int(np.ceil(N * K)), int(K)
def shadow_bound_batch(error, observables, failure_rate=0.01):
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - (np.eye(len(op))*np.trace(op) / 2 ** int(np.log2(op.shape[0]))), ord=np.inf
        )
        ** 2
    )
    N = 0
    for i in tqdm(range(len(observables))):
        obs_ =  pauli_to_matrix(observables[i])
        N = max(34 *shadow_norm(obs_)/ error ** 2, N) 
    return int(np.ceil(N * K)), int(K)    
def moment(n, e_dict_, ham_):
    e = 0
    for i in range(len(ham_.letters)):
        e += ham_.coeffs[i] * e_dict_[ham_.letters[i]]
    return e

def lettersToNums(l):
    # some simple helper func to convert the pauli strings
    # into the form that the shadows use
    numsL = []
    for i in l:
        if(i == "X"):
            numsL += [0]
        elif(i == "Y"):
            numsL += [1]
        elif(i == "Z"):
            numsL += [2]
        else:
            numsL += [3]
    return numsL

# In[16]:
def pauli_to_matrix(os):
    A = [PauliDict_2[os[i]] for i in range(len(os))]
    b = [1]
    for i in range(len(os)):
        b = np.kron(b, A[i])
    return b
def estimate_several_observables(args):
    snap, obs, observables = args
    cexpct = {}
    for i in tqdm(range(len(observables))):
        obs_ = observables[i]
        est = shadows.estimate_shadow_observable_fast((snap,obs),
                                                np.array(lettersToNums(obs_)),int(shadow_size[1]))
        cexpct[obs_] = est
    return cexpct
def ret_circuits(unitary_ids, ansatz):
    circs = []
    shots = []
    unitaries_dict = {}
    for unitary_ in unitary_ids:
        unitary = numsToLetters(unitary_)
        if(unitary in unitaries_dict):
            unitaries_dict[unitary] += 1
        else:
            unitaries_dict[unitary] = 1
    for unitary in unitaries_dict:
        qc = ansatz.copy()
        for i, l in enumerate(unitary):
            if l == "X":
                qc.h(i)
            elif l == "Y":
                qc.rx(np.pi / 2, i)
            qc.measure(i, i)
        circs += [qc]
        shots += [unitaries_dict[unitary]]
    return circs, shots, unitaries_dict
def numsToLetters(t_obs):
    letters = []
    for i in range(len(t_obs)):
        if(t_obs[i] == 0):
            letters += ["X"]
        elif(t_obs[i] == 1):
            letters += ["Y"]
        elif(t_obs[i] == 2):
            letters += ["Z"]
    return "".join(letters)

def qwc_expectation_from_qiskit_job(job, qwc_bases, qwc_shots, paulis):

    cal_results = job.result()
    keys = qwc_bases
    dict_results = {}
    for i in tqdm(range(len(cal_results.results))):
        res = cal_results.results[i].data.counts
        calculations = np.zeros(2**n_qubits)
        for place in res:
            st = bin(int(place, 16))[2:]
            st = str(st)
            add = "0"*int(n_qubits-len(st))
            st = add+st
            #flip for Qiskit endian convention
            basis = stringToBasisState(st[::-1])
            calculations += basis * res[place]/qwc_shots
        dict_results[keys[i]] = calculations
        assert sum(list(res.values())) == qwc_shots, "QWC wrong shots" # this should really never happen

    qwc_expectation_values = {}
    for key in dict_results:
        options = ["ZI"]*n_qubits
        for wire in range(len(key)):
            if(key[wire] == "X"):
                options[wire]= "XI"
            elif(key[wire] == "Y"):
                options[wire]= "YI"
        products = list(product(*options))
        for product_ in products:
            p = "".join(product_)
            if not p in qwc_expectation_values:
                qwc_expectation_values[p] = expectationFromGroupMeasurment(dict_results[key], p)
    return qwc_expectation_values

def get_commuting_cliffords(paulis):
    commuters = []
    for pauli in paulis:
        options = ["ZXY"]*n_qubits
        for wire in range(len(pauli)):
            if(pauli[wire] == "X"):
                options[wire]= "X"
            elif(pauli[wire] == "Y"):
                options[wire]= "Y"
            elif(pauli[wire] == "Z"):
                options[wire]= "Z"
        products = ["".join(w) for w in product(*options)]
        commuters += products
    return list(set(commuters))

circuits_per_job = 20
theta_test = np.genfromtxt(save_dir+"/theta_4_save.txt") # get the ansatz rotation
psi_numerical = circuit_numerical(theta_test)
CUT_OFF = 3 # m < CUT_OFF -> QWC, note not inclusive 
ham_powers = []
NPROCS = 8 # how many cores does the simulator use
qwc_shots = 1024 # how many shots for each QWC base
mix_qwc_shots = int(qwc_shots * 1.38)
trial_ks = [1,2,3,4,5]

for i in range(2*K):
    with open("saved_hamiltonians/linear_"+str(n_qubits)+"_"+str(i)+".json") as fp:
        dict_ = json.load(fp)
    hami = ham.Hamiltonian(list(dict_.keys()), [complex(s) for s in list(dict_.values())])
    ham_powers += [hami]
ham_qm_powers = ham_powers
ham_qm = ham_qm_powers[1] # name for H

total_paulis = []
for m in range(len(ham_powers)):
        total_paulis += ham_powers[m].letters
        total_paulis = list(set(total_paulis))
# now we calculate the bases needed for pure QWC
total_ham = ham.Hamiltonian(total_paulis,[1]*len(total_paulis))
total_qwc_bases = total_ham.grouping()

# set the budget
budget = qwc_shots * len(total_qwc_bases)

provider = IBMQ.load_account()
print(print(IBMQ.providers()))
provider = IBMQ.get_provider("ibm-q-pnnl")
quito = provider.get_backend('ibmq_quito')
noise_model = NoiseModel.from_backend(quito)

from qiskit import Aer
for round_index in range(n_rounds): # run several experiments for each method
    print("PERCENT MIX: ", 100 * mix_qwc_shots/qwc_shots, "%")
    from qiskit import Aer
    qc = circuit(theta_test)
    print("Pure QWC with ", len(total_qwc_bases))

    ### ======== ###
    ### PURE QWC ###
    ### ======== ###

    # split QWC bases not to overload processors and get progress bar
    splits = np.array_split(total_qwc_bases, NPROCS)
    circs = []
    for i in range(len(total_qwc_bases)):
        base = total_qwc_bases[i]
        qc = circuit(theta_test)
        for i in range(len(base)):

            #rotate qubits into the correct computational basis
            if base[i] == "X":
                qc.h(i)
            elif base[i] == "Y":
                qc.rx(np.pi/2,i)

            qc.measure(i,i)
        circs += [qc]

    qbackend = AerSimulator()
    # Set executor and max_job_size
    exc = ThreadPoolExecutor(max_workers=NPROCS)
    qbackend.set_options(executor=exc)
    qbackend.set_options(max_job_size=NPROCS)
    qbackend.set_options(noise_model=noise_model)
    job = qbackend.run(circs, shots=qwc_shots)

    print("Calculating QWC Expectation Values")
    pure_qwc_expectations = qwc_expectation_from_qiskit_job(job, total_qwc_bases, qwc_shots, total_paulis)   
    pure_qwc_moments = {0: 1.0}
    for i in range(2*K):
        power_ham = ham_qm_powers[i]
        pure_qwc_moments[i] = moment(i, pure_qwc_expectations, power_ham)
    pure_qwc_moments = {0: 1.0}
    for i in range(1,2*K):
        power_ham = ham_qm_powers[i]
        pure_qwc_moments[i] = moment(i, pure_qwc_expectations, power_ham)

    pure_qwc_est_es = [comp_energy(pure_qwc_moments, i) for i in trial_ks]
    pure_qwc_shots = len(total_qwc_bases) * qwc_shots

    data_save = {}
    data_save["qwc_energy"] = pure_qwc_est_es
    data_save["qwc_shots"] = pure_qwc_shots
    data_save["qwc_moments"] = [str(s) for s in list(pure_qwc_moments.values())]


    with open("pure_qwc_expectations/"+str(round_index)+".json", "w") as fp:
        json.dump(pure_qwc_expectations, fp)
    print("Starting Mixing")

    ### ======== ###
    ###  MIXING  ###
    ### ======== ###

    # calculate the qwc bases needed for first part of mixing
    mix_qwc_paulis = []
    for i in range(2*K):
        if(i < CUT_OFF): mix_qwc_paulis += ham_qm.power(i).letters
    mix_qwc_paulis = list(set(mix_qwc_paulis))
    mix_qwc_bases = ham.Hamiltonian(mix_qwc_paulis, np.ones(len(mix_qwc_paulis))).grouping()
    print("Calculating ", len(mix_qwc_bases), " qwc bases for mixing")
    circs = []
    for base in mix_qwc_bases:
        qc = circuit(theta_test)
        for i in range(len(base)):
            #rotate qubits into the correct computational basis
            if base[i] == "X":
                qc.h(i)
            elif base[i] == "Y":
                qc.rx(np.pi/2,i)
            qc.measure(i,i)
        circs += [qc]
    print("Executing QWC Circuits ", len(circs))
    qbackend = AerSimulator()
    # Set executor and max_job_size
    exc = ThreadPoolExecutor(max_workers=NPROCS)
    qbackend.set_options(executor=exc)
    qbackend.set_options(max_job_size=NPROCS)
    qbackend.set_options(noise_model=noise_model)
    job = qm.execute(circs, shots=mix_qwc_shots, backend=qbackend)
    mix_qwc_expectations = qwc_expectation_from_qiskit_job(job, mix_qwc_bases, mix_qwc_shots, mix_qwc_paulis)

    print("Running CS Portion of Mixing")

    # calculate the remaining paulis
    cs_paulis = list(set(total_paulis).difference(set(list(mix_qwc_expectations.keys()))))
    commuting_cliffords = get_commuting_cliffords(cs_paulis)     
    circuits = []
    for unitary in commuting_cliffords:
        qc = circuit(theta_test).copy()
        for i, l in enumerate(unitary):
            if l == "X":
                qc.h(i)
            elif l == "Y":
                qc.rx(np.pi / 2, i)
            qc.measure(i, i)
        circuits += [qc]
    shots_ = int((budget - mix_qwc_shots * len(mix_qwc_bases))/len(commuting_cliffords))
    print("Submitting Job With ", len(circuits), " circuits and ", shots_, " shots per circuit")
    counts = []
    array_ind = np.array_split(list(range(len(circuits))), np.ceil(len(circuits)/circuits_per_job))
    print(array_ind)
    qbackend = AerSimulator()
    # Set executor and max_job_size
    exc = ThreadPoolExecutor(max_workers=NPROCS)
    qbackend.set_options(executor=exc)
    qbackend.set_options(noise_model=noise_model)
    qbackend.set_options(max_job_size=NPROCS)
    for arr in array_ind:
        c_circuits = [circuits[i] for i in arr]
        result = qbackend.run(c_circuits, shots=shots_).result()
        counts += result.get_counts()
    snap = []
    # format results for CS estimator
    for count in counts:
        s = 0
        for key in count:
            s += count[key]
            key_ = key[::-1]
            key_ = [[-2 * (float(i) - 0.5) for i in key_] for k in range(count[key])]
            snap += key_
        assert s == shots_

    unitary_ids_formatted = []
    for unitary in commuting_cliffords:
        u_mat = [lettersToNums(unitary)]*shots_
        unitary_ids_formatted += u_mat

    unitary_ids_formatted = np.array(unitary_ids_formatted)
    snap = np.array(snap)
    permute = np.random.permutation(list(range(snap.shape[0]))) # CS requires us to have the results in a random order
                                                                # if we use median of means estimation
    snap = snap[permute]
    unitary_ids_formatted = unitary_ids_formatted[permute]
    print(snap.shape)

    # given the shadow (snap, unitary_ids_formatted), calculate the remaining expectation values
    mix_cs_expectations = {}
    shadow = (snap, unitary_ids_formatted)
    for obs_ in cs_paulis:
        est = shadows.estimate_shadow_observable_mix_modification(shadow, np.array(lettersToNums(obs_)), 1) # K = 1, no median of means
        mix_cs_expectations[obs_] = est
    mix_expectations = mix_qwc_expectations.copy()
    mix_expectations.update(mix_cs_expectations)
    

    # calculate the moments given these expectations
    mix_moments = {0: 1.0}
    for i in range(2*K):
        power_ham = ham_qm_powers[i]
        mix_moments[i] = moment(i, mix_expectations, power_ham)
    est_es = [comp_energy(mix_moments, i) for i in trial_ks]

    data_save["mix_energy"] = est_es
    data_save["mix_shots"] = mix_qwc_shots * len(mix_qwc_bases) + snap.shape[0]
    data_save["mix_moments"] = [str(s) for s in list(mix_moments.values())]
    with open("mix_expectations/"+str(round_index)+".json","w") as fp:
        json.dump(mix_expectations, fp)

    
    ### ======= ###
    ### Pure CS ###
    ### ======= ###

    commuting_cliffords = get_commuting_cliffords(total_paulis)  
    circuits = []
    for unitary in commuting_cliffords:
        qc = circuit(theta_test).copy()
        for i, l in enumerate(unitary):
            if l == "X":
                qc.h(i)
            elif l == "Y":
                qc.rx(np.pi / 2, i)
            qc.measure(i, i)
        circuits += [qc]
    shots_ = int(budget /len(commuting_cliffords))
    print("Submitting Job With ", len(circuits), " circuits and ", shots_, " shots per circuit")
    counts = []
    array_ind = np.array_split(list(range(len(circuits))), int(len(circuits)/circuits_per_job))
    qbackend = AerSimulator()
    # Set executor and max_job_size
    exc = ThreadPoolExecutor(max_workers=NPROCS)
    qbackend.set_options(executor=exc)
    qbackend.set_options(max_job_size=NPROCS)
    qbackend.set_options(noise_model=noise_model)
    for arr in tqdm(array_ind):
        c_circuits = [circuits[i] for i in arr]
        result = qbackend.run(c_circuits,shots=shots_).result()
        counts += result.get_counts()
    snap = []
    # format for CS estimator
    for count in counts:
        s = 0
        for key in count:
            s += count[key]
            key_ = key[::-1]
            key_ = [[-2 * (float(i) - 0.5) for i in key_] for k in range(count[key])]
            snap += key_
        assert s == shots_

    unitary_ids_formatted = []
    for unitary in commuting_cliffords:
        u_mat = [lettersToNums(unitary)]*shots_
        unitary_ids_formatted += u_mat
    unitary_ids_formatted = np.array(unitary_ids_formatted)
    snap = np.array(snap)
    permute = np.random.permutation(list(range(snap.shape[0]))) # again we might need the shadow scrambled
    snap = snap[permute]
    unitary_ids_formatted = unitary_ids_formatted[permute]
    print(snap.shape)
    print(unitary_ids_formatted.shape)
    pure_cs_expectations = {}
    shadow = (snap, unitary_ids_formatted)
    for obs_ in tqdm(total_paulis):
        est = shadows.estimate_shadow_observable_fast(shadow, np.array(lettersToNums(obs_)), 17) # K = 17, using median of means
        pure_cs_expectations[obs_] = est

    pure_cs_moments = {0: 1.0}
    for i in range(2*K):
        power_ham = ham_qm_powers[i]
        pure_cs_moments[i] = moment(i, pure_cs_expectations, power_ham)
    pure_cs_est_es = [comp_energy(pure_cs_moments, i) for i in trial_ks]

    data_save["cs_energy"] = pure_cs_est_es
    data_save["cs_shots"] = snap.shape[0]
    data_save["cs_moments"] = [str(s) for s in list(pure_cs_moments.values())]

    with open("pure_cs_expectations/"+str(round_index)+".json", "w") as fp:
        json.dump(pure_cs_expectations, fp)

    with open(save_dir + "/results/" + str(round_index) + ".json","w") as fp:
        json.dump(data_save, fp)

    print("  ==   ", 100*round_index/n_rounds, "  ==  % Complete")
