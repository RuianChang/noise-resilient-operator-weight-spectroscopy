import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

# Single-qubit Clifford gates
Id  = np.array([[1.,0],[0,1]], dtype=complex)
s_x = np.array([[0,1.],[1,0]], dtype=complex)
s_y = np.array([[0,-1j],[1j,0]], dtype=complex)
s_z = np.array([[1.,0],[0,-1]], dtype=complex)

X1 = (Id+1j*s_x)/np.sqrt(2)
X2 = (Id-1j*s_x)/np.sqrt(2)
Y1 = (Id+1j*s_y)/np.sqrt(2)
Y2 = (Id-1j*s_y)/np.sqrt(2)
Z1 = (Id+1j*s_z)/np.sqrt(2)
Z2 = (Id-1j*s_z)/np.sqrt(2)

H1 = (s_x+s_z)/np.sqrt(2)
H2 = (s_x-s_z)/np.sqrt(2)
H3 = (s_x+s_y)/np.sqrt(2)
H4 = (s_x-s_y)/np.sqrt(2)
H5 = (s_y+s_z)/np.sqrt(2)
H6 = (s_y-s_z)/np.sqrt(2)

L1 = (Id+1j*(s_x+s_y+s_z))/2
L2 = (Id+1j*(s_x+s_y-s_z))/2
L3 = (Id+1j*(s_x-s_y+s_z))/2
L4 = (Id+1j*(-s_x+s_y+s_z))/2
L5 = (Id+1j*(s_x-s_y-s_z))/2
L6 = (Id+1j*(-s_x+s_y-s_z))/2
L7 = (Id+1j*(-s_x-s_y+s_z))/2
L8 = (Id+1j*(-s_x-s_y-s_z))/2

# Two-qubit Clifford gates
Id_2 = np.array([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=complex)
SWAP = np.array([[1.,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
iSWAP = np.array([[1.,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
CNOT = np.array([[1.,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

T_gate = np.array([[1, 0],
              [0, np.exp(1j*np.pi/4)]], dtype=complex)

def random_two_single_qubit_cliffords(replace=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    single_qubit_cliffords = [
        Id,
        X1, X2,
        Y1, Y2,
        Z1, Z2,
        H1, H2, H3, H4, H5, H6,
        L1, L2, L3, L4, L5, L6, L7
    ]

    idx = rng.choice(len(single_qubit_cliffords), size=2, replace=replace)
    return single_qubit_cliffords[idx[0]], single_qubit_cliffords[idx[1]]

def random_two_qubit_clifford(rng=None):
    if rng is None:
        rng = np.random.default_rng()

    two_qubit_cliffords = [Id_2, SWAP, iSWAP, CNOT]
    return two_qubit_cliffords[rng.integers(len(two_qubit_cliffords))]

def random_pauli(rng=None):
    if rng is None:
        rng = np.random.default_rng()

    paulis = [Id, s_x, s_y, s_z]
    return paulis[rng.integers(len(paulis))]

# Haar random unitary (complex)
def haar_rand(d):
    A = (np.random.randn(d,d) + 1j*np.random.randn(d,d))/np.sqrt(2)
    Q, R = np.linalg.qr(A)
    diag = np.diag(R)
    Q = Q @ np.diag(diag/np.abs(diag))
    return Q

def build_single_site_paulis(L):
    X_list, Y_list, Z_list, T_list = [], [], [], []
    for i in range(L):
        X = np.kron(np.eye(2**i), np.kron(s_x, np.eye(2**(L-i-1))))
        Y = np.kron(np.eye(2**i), np.kron(s_y, np.eye(2**(L-i-1))))
        Z = np.kron(np.eye(2**i), np.kron(s_z, np.eye(2**(L-i-1))))
        T = np.kron(np.eye(2**i), np.kron(T_gate, np.eye(2**(L-i-1))))
        X_list.append(X); Y_list.append(Y); Z_list.append(Z); T_list.append(T)
    return X_list, Y_list, Z_list, T_list
    
def random_pauli_and_T(P, L, i, pT, X_list, Y_list, Z_list, T_list):
    """Uniform Pauli kick: sigma in {I,X,Y,Z} with prob 1/4. and T gate with prob pT."""

    k = np.random.randint(4)  # 0,1,2,3 equally likely
    if k == 0:
        sigma = np.eye(2**L)  # Identity
    elif k == 1:
        sigma = X_list[i]
    elif k == 2:
        sigma = Y_list[i]
    else:
        sigma = Z_list[i]
    
    if np.random.rand() < pT: # apply T gate with probability pT
        deph = sigma @ T_list[i]
        P = deph.conj().T @ P @ deph
    else:
        P = sigma.conj().T @ P @ sigma # No T gate applied, just the random Pauli kick
    return P

def apply_depolarizing_channel(P, i, r, X_list, Y_list, Z_list):
    """
    EXACT deterministic depolarizing channel:
      P -> a P + b (X P X + Y P Y + Z P Z)
    a = (1 + 3 e^{-r})/4, b = (1 - e^{-r})/4
    """
    e = np.exp(-r)
    a = (1 + 3*e) / 4
    b = (1 - e) / 4
    X = X_list[i]; Y = Y_list[i]; Z = Z_list[i]
    return a * P + b * (X @ P @ X + Y @ P @ Y + Z @ P @ Z)


def make_frozen_clifford_layers(L, Tstep):
    """Sample Clifford gates once and freeze them.

    Returns
    -------
    even_cliffords : list[list[np.ndarray]]
        even_cliffords[t][k] is the 4x4 Clifford acting on pair (2k,2k+1) at time t.
    odd_cliffords : list[list[np.ndarray]]
        odd_cliffords[t][k] is the 4x4 Clifford acting on pair (2k+1,2k+2) at time t.
    """
    even_pairs = list(range(0, L-1, 2))
    odd_pairs  = list(range(1, L-1, 2))

    even_cliffords, odd_cliffords = [], []
    for _t in range(Tstep):
        even_cliffords_t = []
        for _ in even_pairs:
            U1, U2 = random_two_single_qubit_cliffords()
            U3 = random_two_qubit_clifford()
            clifford_2q = np.kron(U1,U2) @ U3
            even_cliffords_t.append(clifford_2q)
        even_cliffords.append(even_cliffords_t)

        odd_cliffords_t = []
        for _ in odd_pairs:
            U1, U2 = random_two_single_qubit_cliffords()
            U3 = random_two_qubit_clifford()
            clifford_2q = np.kron(U1,U2) @ U3
            odd_cliffords_t.append(clifford_2q)
        odd_cliffords.append(odd_cliffords_t)

    return even_cliffords, odd_cliffords


def layer_even_frozen_clifford_with_magic(P, L, r, frozen_even_2q, pT, X_list, Y_list, Z_list, T_list):
    """One even layer evolution: (frozen Clifford) + (random magic sampling) + depolarizing noise.
    """
    U_even = 1 # start with scalar 1, then kron with each 2-qubit Clifford
    for U_2q in frozen_even_2q:
        U_even = np.kron(U_even, U_2q)
    P = (U_even.conj().T) @ P @ U_even

    # random Paulis, T gates, deterministic depolarizing on all sites
    for i in range(L):
        P = random_pauli_and_T(P, L, i, pT, X_list, Y_list, Z_list, T_list)       
        P = apply_depolarizing_channel(P, i, r, X_list, Y_list, Z_list)
    return P


def layer_odd_frozen_clifford_with_magic(P, L, r, frozen_odd_2q, pT, X_list, Y_list, Z_list, T_list):
    """One odd layer evolution: (frozen Clifford) + (random magic sampling) + depolarizing noise.
    """
    U_odd = 1 # start with scalar 1, then kron with each 2-qubit Clifford
    for U_2q in frozen_odd_2q:
        U_odd = np.kron(U_odd, U_2q)
    U_odd = np.kron(Id, np.kron(U_odd, Id))
    P = (U_odd.conj().T) @ P @ U_odd

    # random Paulis, T gates, deterministic depolarizing on internal sites only
    for i in range(1, L-1):
        P = random_pauli_and_T(P, L, i, pT, X_list, Y_list, Z_list, T_list)       
        P = apply_depolarizing_channel(P, i, r, X_list, Y_list, Z_list)
    return P

# ---------------------------
# main
# ---------------------------
L = 4
N_layer = 4  # Each layer includes even and odd, so 2*N_layer unitary layers in total.
s = L // 2

pT_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # magic insertion probability
Ns = 150  # average over CIRCUIT REALIZATIONS (magic sampling) at fixed frozen Clifford circuit

# Precompute full-system single-site Paulis
X_list, Y_list, Z_list, T_list = build_single_site_paulis(L)

# Freeze the Clifford circuit ONCE (shared across all pT and r)
frozen_even, frozen_odd = make_frozen_clifford_layers(L, N_layer)

# Scan noise r (same as your original)
n_points = 500 # ~2 * (2*N_layer)**2
min_noise = 0.003
max_noise = 0.90
min_exp = np.exp(-2*max_noise)
max_exp = np.exp(-2*min_noise)
noise_range = -0.5*np.log(np.linspace(min_exp, max_exp, n_points))

corr_ave = []   # will become shape (n_pT, n_noise)

for pT in pT_list:
    corr_vs_r = []
    print('\no', end='', flush=True)

    for i, r in enumerate(noise_range):
        if i % 50 == 0:
            print('.', end='', flush=True)

        corr_accum = 0.0 + 0.0j

        for _n in range(Ns):
            # initial operator
            P = np.kron(np.eye(2**(s-1)), np.kron(s_z, np.eye(2**(L-s)))) # Z operator in the middle, identity elsewhere
            
            for t in range(N_layer):
                P = layer_even_frozen_clifford_with_magic(P, L, r, frozen_even[t], pT, X_list, Y_list, Z_list, T_list)
                P = layer_odd_frozen_clifford_with_magic(P, L, r, frozen_odd[t],  pT, X_list, Y_list, Z_list, T_list)

            correlation = np.trace(P @ P) / (2**L)
            corr_accum += correlation

        corr_vs_r.append(corr_accum / Ns)

    corr_ave.append(corr_vs_r)

corr_ave = np.real(np.array(corr_ave))

X = np.exp(-2*noise_range)

for i, pT in enumerate(pT_list):
    plt.plot(X, corr_ave[i], label=rf"$p_T={pT}$")

plt.xlabel(r"$e^{-2\gamma}$")
plt.ylabel("Correlation")
plt.legend()
plt.show()