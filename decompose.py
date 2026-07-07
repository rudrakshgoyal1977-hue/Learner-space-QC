from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
import rotation

def num_qubits(N: int) -> int:
    return int(np.round(np.log2(N)))

@dataclass
class TwoLevel:
    size: int
    level0: int
    level1: int
    unitary: np.ndarray

    def to_unitary(self) -> np.ndarray:
        U = np.eye(self.size, dtype=np.complex128)
        U[self.level0, self.level0] = self.unitary[0, 0]
        U[self.level0, self.level1] = self.unitary[0, 1]
        U[self.level1, self.level0] = self.unitary[1, 0]
        U[self.level1, self.level1] = self.unitary[1, 1]
        return U

@dataclass
class SingleQubitGate:
    n: int
    qubit: int
    unitary: np.ndarray

    def to_unitary(self) -> np.ndarray:
        N = 2**self.n
        U = np.zeros((N, N), dtype=np.complex128)
        bit = 1 << self.qubit
        for i in range(N):
            if not (i & bit):
                j = i | bit
                U[i, i] = self.unitary[0, 0]
                U[i, j] = self.unitary[0, 1]
                U[j, i] = self.unitary[1, 0]
                U[j, j] = self.unitary[1, 1]
        return U

@dataclass
class ControlledU:
    n: int
    target: int
    unitary: np.ndarray

    def to_unitary(self) -> np.ndarray:
        N = 2**self.n
        U = np.eye(N, dtype=np.complex128)
        mask = ((1 << self.n) - 1) ^ (1 << self.target)
        idx0, idx1 = mask, mask | (1 << self.target)
        U[idx0, idx0] = self.unitary[0, 0]
        U[idx0, idx1] = self.unitary[0, 1]
        U[idx1, idx0] = self.unitary[1, 0]
        U[idx1, idx1] = self.unitary[1, 1]
        return U

@dataclass
class CU:
    n: int
    control: int
    target: int
    unitary: np.ndarray

    def to_unitary(self) -> np.ndarray:
        N = 2**self.n
        U = np.eye(N, dtype=np.complex128)
        c_bit, t_bit = 1 << self.control, 1 << self.target
        for i in range(N):
            if (i & c_bit) and not (i & t_bit):
                j = i | t_bit
                U[i, i] = self.unitary[0, 0]
                U[i, j] = self.unitary[0, 1]
                U[j, i] = self.unitary[1, 0]
                U[j, j] = self.unitary[1, 1]
        return U

@dataclass
class CNOT:
    n: int
    control: int
    target: int

    def to_unitary(self) -> np.ndarray:
        N = 2**self.n
        U = np.eye(N, dtype=np.complex128)
        c_bit, t_bit = 1 << self.control, 1 << self.target
        for i in range(N):
            if (i & c_bit) and not (i & t_bit):
                j = i | t_bit
                U[i, i], U[i, j] = 0, 1
                U[j, i], U[j, j] = 1, 0
        return U

@dataclass
class Swap:
    target: int
    control_vals: list[bool]

Gate = Union[TwoLevel, SingleQubitGate, ControlledU, CU, CNOT]
Circuit = list
TwoLevels = list

def circuit_to_unitary(circuit: Circuit) -> np.ndarray:
    if not circuit: return None
    try:
        N = 2**circuit[0].n
    except AttributeError:
        N = circuit[0].size
    U = np.eye(N, dtype=np.complex128)
    for gate in circuit:
        U = gate.to_unitary() @ U
    return U

def to_circuit(two_levels: TwoLevels) -> Circuit:
    return list(two_levels)

def error_up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    overlap = np.trace(b.conj().T @ a)
    phase = np.angle(overlap)
    b_aligned = b * np.exp(1j * phase)
    return np.linalg.norm(a - b_aligned)

def align(x: complex, y: complex, norm: float) -> np.ndarray:
    if norm < 1e-12: return np.eye(2, dtype=np.complex128)
    return np.array([[np.conj(x), np.conj(y)], [-y, x]]) / norm

def decompose_vector(vec: np.ndarray) -> TwoLevels:
    tls = []
    n_dim = len(vec)
    current_vec = vec.copy()
    for i in range(n_dim - 1, 0, -1):
        x, y = current_vec[i-1], current_vec[i]
        norm_val = np.sqrt(np.abs(x)**2 + np.abs(y)**2)
        if np.abs(y) > 1e-12:
            u2 = align(x, y, norm_val)
            tls.append(TwoLevel(size=n_dim, level0=i-1, level1=i, unitary=u2))
            current_vec[i-1] = norm_val
            current_vec[i] = 0
    return tls

def expand_twolevels(input: TwoLevels, n: int) -> TwoLevels:
    return [TwoLevel(size=n, level0=tl.level0 + (n - tl.size), level1=tl.level1 + (n - tl.size), unitary=tl.unitary) for tl in input]

def two_levels_to_unitary(two_levels: TwoLevels) -> np.ndarray:
    if not two_levels: return None
    U = np.eye(two_levels[0].size, dtype=np.complex128)
    for tl in two_levels: U = tl.to_unitary() @ U
    return U

def adjoint_twolevel(tl: TwoLevel) -> TwoLevel:
    return TwoLevel(size=tl.size, level0=tl.level0, level1=tl.level1, unitary=tl.unitary.conj().T)

def adjoint_twolevels(two_levels: TwoLevels) -> TwoLevels:
    return [adjoint_twolevel(tl) for tl in reversed(two_levels)]

def decompose_unitary(u: np.ndarray) -> TwoLevels:
    N = u.shape[0]
    tls = []
    current_u = u.copy()
    for i in range(N - 1):
        col = current_u[i:, i]
        sub_tls = decompose_vector(col)
        expanded = expand_twolevels(sub_tls, N)
        for tl in expanded:
            current_u = tl.to_unitary() @ current_u
            tls.append(tl)
            
    phase = current_u[N-1, N-1]
    if not np.isclose(phase, 1.0):
        u_phase = np.array([[1, 0], [0, np.conj(phase)]])
        tl = TwoLevel(size=N, level0=N-2, level1=N-1, unitary=u_phase)
        tls.append(tl)
    return tls

def twolevel_decomposition(u: np.ndarray) -> TwoLevels:
    return adjoint_twolevels(decompose_unitary(u))

@dataclass
class ABC:
    alpha: float
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray

def abc_decompose(u: np.ndarray) -> ABC:
    alpha, beta, gamma, delta = rotation.euler_angles_zyz(u)
    A = rotation.Rz(beta) @ rotation.Ry(gamma / 2.0)
    B = rotation.Ry(-gamma / 2.0) @ rotation.Rz(-(delta + beta) / 2.0)
    C = rotation.Rz((delta - beta) / 2.0)
    return ABC(alpha=alpha, A=A, B=B, C=C)

def abc_reconstruct(d: ABC) -> np.ndarray:
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return np.exp(1j * d.alpha) * (d.A @ X @ d.B @ X @ d.C)

def gray_code(tl: TwoLevel) -> list[Swap]:
    diff = tl.level0 ^ tl.level1
    swaps = []
    curr = tl.level0
    n = num_qubits(tl.size)
    for i in range(n):
        if diff & (1 << i):
            target = i
            c_vals = [bool((curr >> j) & 1) for j in range(n)]
            swaps.append(Swap(target=target, control_vals=c_vals))
            curr ^= (1 << i)
    return swaps

def decompose_swap(swap: Swap) -> Circuit:
    n = len(swap.control_vals)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return controlled_circuit(n, swap.target, swap.control_vals, X)

def controlled_circuit(n: int, target: int, control_vals: list[bool], unitary: np.ndarray) -> Circuit:
    circ = []
    X_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    for i, bit in enumerate(control_vals):
        if i != target and not bit:
            circ.append(SingleQubitGate(n, i, X_gate))
    circ.append(ControlledU(n, target, unitary))
    for i, bit in enumerate(control_vals):
        if i != target and not bit:
            circ.append(SingleQubitGate(n, i, X_gate))
    return circ

def decompose_twolevel(tl: TwoLevel) -> Circuit:
    swaps = gray_code(tl)
    if not swaps: return []
    circ = []
    for swap in swaps[:-1]:
        circ.extend(decompose_swap(swap))
    last_swap = swaps[-1]
    circ.extend(controlled_circuit(num_qubits(tl.size), last_swap.target, last_swap.control_vals, tl.unitary))
    for swap in reversed(swaps[:-1]):
        circ.extend(decompose_swap(swap))
    return circ

def decompose_controlled(n: int, controls: list[int], target: int, u: np.ndarray) -> Circuit:
    if not controls:
        return [SingleQubitGate(n, target, u)]
    if len(controls) == 1:
        if np.allclose(u, np.array([[0, 1], [1, 0]])): return [CNOT(n, controls[0], target)]
        return [CU(n, controls[0], target, u)]
        
    V = rotation.unitary2_sqrt(u)
    V_dag = V.conj().T
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pivot = controls[-1]
    rem_controls = controls[:-1]
    
    c = []
    c.extend(decompose_controlled(n, [pivot], target, V))
    c.extend(decompose_controlled(n, rem_controls, pivot, X))
    c.extend(decompose_controlled(n, [pivot], target, V_dag))
    c.extend(decompose_controlled(n, rem_controls, pivot, X))
    c.extend(decompose_controlled(n, rem_controls, target, V))
    return c

def decompose_controlledU(g: ControlledU) -> Circuit:
    controls = [i for i in range(g.n) if i != g.target]
    return decompose_controlled(g.n, controls, g.target, g.unitary)

def decompose_cu(g: CU) -> Circuit:
    abc = abc_decompose(g.unitary)
    c = []
    c.append(SingleQubitGate(g.n, g.target, abc.C))
    c.append(CNOT(g.n, g.control, g.target))
    c.append(SingleQubitGate(g.n, g.target, abc.B))
    c.append(CNOT(g.n, g.control, g.target))
    c.append(SingleQubitGate(g.n, g.target, abc.A))
    
    phase_gate = np.array([[1, 0], [0, np.exp(1j * abc.alpha)]], dtype=np.complex128)
    c.append(SingleQubitGate(g.n, g.control, phase_gate))
    return c

def decompose_to_basis(u: np.ndarray) -> Circuit:
    tls = twolevel_decomposition(u)
    c1 = []
    for tl in tls: c1.extend(decompose_twolevel(tl))
    c2 = []
    for gate in c1:
        if isinstance(gate, ControlledU): c2.extend(decompose_controlledU(gate))
        else: c2.append(gate)
    c3 = []
    for gate in c2:
        if isinstance(gate, CU): c3.extend(decompose_cu(gate))
        else: c3.append(gate)
    return c3

def ht_gates(n: int, qubit: int, word: str) -> Circuit:
    c = []
    H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T_gate = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
    for char in reversed(word):
        g = H_gate if char == 'H' else T_gate
        c.append(SingleQubitGate(n, qubit, g))
    return c

def decompose_to_ht(u: np.ndarray, error: float) -> Circuit:
    basis_circ = decompose_to_basis(u)
    final_circ = []
    for gate in basis_circ:
        if isinstance(gate, SingleQubitGate):
            word = rotation.approximate_in_ht(gate.unitary, error)
            final_circ.extend(ht_gates(gate.n, gate.qubit, word))
        else:
            final_circ.append(gate)
    return final_circ