import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi

class Bloch:
    alpha: float
    n: np.ndarray
    theta: float
    
    def __init__(self, alpha, n, theta):
        self.alpha = alpha
        self.n = n
        self.theta = theta


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    det = np.linalg.det(g)
    alpha = np.angle(det) / 2.0
    s = g * np.exp(-1j * alpha)
    
    cos_half = np.real((s[0, 0] + s[1, 1]) / 2.0)
    cos_half = np.clip(cos_half, -1.0, 1.0)
    theta = 2.0 * np.arccos(cos_half)
    
    if np.isclose(theta, 0.0, atol=1e-8):
        return Bloch(alpha=alpha, n=np.array([0.0, 0.0, 1.0]), theta=0.0)
        
    sin_half = np.sin(theta / 2.0)
    nx = -np.imag(s[0, 1] + s[1, 0]) / (2.0 * sin_half)
    ny = np.real(s[0, 1] - s[1, 0]) / (2.0 * sin_half)
    nz = -np.imag(s[0, 0] - s[1, 1]) / (2.0 * sin_half)
    
    n = np.array([nx, ny, nz], dtype=float)
    n /= np.linalg.norm(n)
    return Bloch(alpha=alpha, n=n, theta=theta)


def expand_word(word: list[int]) -> str:
    """Flatten an alternating exponent list into a literal string of 'H'/'T' gates."""
    res = []
    for i, count in enumerate(word):
        char = 'T' if i % 2 == 0 else 'H'
        res.append(char * count)
    return "".join(res)

def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string."""
    U = np.eye(2, dtype=DTYPE)
    T_gate = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=DTYPE)
    for char in gates:
        if char == 'H': U = U @ H
        elif char == 'T': U = U @ T_gate
    return U

# Compute dynamic n1, n2 from the provided M1 and M2 words.
M1_WORD = [7, 1, 1, 1]
M2_WORD = [2, 1, 1, 1, 6, 1, 7, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 7, 1, 6]
M1_STR = expand_word(M1_WORD)
M2_STR = expand_word(M2_WORD)

b1 = to_bloch(gates_to_unitary(M1_STR))
b2 = to_bloch(gates_to_unitary(M2_STR))

n1 = -b1.n
n2 = -b2.n
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)

def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor rotation into u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)."""
    R_frame = np.column_stack((a3, a2, a1)) # maps (X, Y, Z) to (a3, a2, a1)
    n_prime = R_frame.T @ b.n
    b_prime = Bloch(alpha=b.alpha, n=n_prime, theta=b.theta)
    u_prime = from_axis_angle(b_prime)
    
    alpha_zyz, beta_zyz, gamma_zyz, delta_zyz = euler_angles_zyz(u_prime)
    return (-delta_zyz % TWO_PI, -gamma_zyz % TWO_PI, -beta_zyz % TWO_PI, alpha_zyz)

def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find k such that (k * LAMBDA_PI) mod 2*pi ~= angle."""
    angle = angle % TWO_PI
    k = 1
    while True:
        val = (k * LAMBDA_PI) % TWO_PI
        dist = min(abs(val - angle), TWO_PI - abs(val - angle))
        if dist <= tolerance:
            return k
        k += 1

def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate 2x2 unitary u as M1^k * M2^l * M1^m."""
    alpha, beta, gamma, _ = n1n2n1_angles(to_bloch(u))
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)
    return (k, l, m)

def from_axis_angle(b: Bloch) -> np.ndarray:
    """Build a 2x2 unitary from its Bloch form."""
    I = np.eye(2, dtype=DTYPE)
    X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
    Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
    Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)
    n_dot_sigma = b.n[0]*X + b.n[1]*Y + b.n[2]*Z
    return np.exp(1j * b.alpha) * (np.cos(b.theta/2.0) * I - 1j * np.sin(b.theta/2.0) * n_dot_sigma)

def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=DTYPE)

def Ry(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=DTYPE)

def euler_angles_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    """ZYZ Euler decomposition of a 2x2 unitary."""
    det = np.linalg.det(u)
    alpha = np.angle(det) / 2.0
    S = np.exp(-1j * alpha) * u
    cos_g = np.abs(S[0, 0])
    sin_g = np.abs(S[1, 0])
    gamma = 2.0 * np.arctan2(sin_g, cos_g)
    
    if np.isclose(sin_g, 0.0, atol=1e-8):
        beta = -2.0 * np.angle(S[0, 0])
        delta = 0.0
    else:
        beta_plus_delta = -2.0 * np.angle(S[0, 0])
        beta_minus_delta = 2.0 * np.angle(S[1, 0])
        beta = (beta_plus_delta + beta_minus_delta) / 2.0
        delta = (beta_plus_delta - beta_minus_delta) / 2.0
        
    return (alpha, beta % TWO_PI, gamma % TWO_PI, delta % TWO_PI)

def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    """Principal square root of a 2x2 unitary."""
    b = to_bloch(u)
    return from_axis_angle(Bloch(alpha=b.alpha/2.0, n=b.n, theta=b.theta/2.0))

def invert_gates(gates: str) -> str:
    """Inverse of a flat H/T word."""
    return "".join('H' if c == 'H' else 'T'*7 for c in reversed(gates))

def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word."""
    if k == 0: return ""
    elif k > 0: return base * k
    else: return invert_gates(base) * (-k)

def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary `u` by a flat H/T word."""
    k, l, m = decompose_2x2(u, error)
    return power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m)
