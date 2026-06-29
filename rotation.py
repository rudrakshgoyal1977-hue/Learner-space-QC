import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    det_g = np.linalg.det(g)
    alpha = np.angle(det_g) / 2.0
    
    g_tilde = g * np.exp(-1j * alpha)
    
    trace_val = np.real(np.trace(g_tilde))
    theta = 2.0 * np.arccos(np.clip(trace_val / 2.0, -1.0, 1.0))
    
    if np.isclose(theta, 0.0) or np.isclose(theta, TWO_PI):
        n = np.array([1.0, 0.0, 0.0])
    else:
        X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
        Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
        Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)
        
        sin_half_theta = np.sin(theta / 2.0)
        
        nx = np.real(1j * np.trace(X @ g_tilde) / (2.0 * sin_half_theta))
        ny = np.real(1j * np.trace(Y @ g_tilde) / (2.0 * sin_half_theta))
        nz = np.real(1j * np.trace(Z @ g_tilde) / (2.0 * sin_half_theta))
        
        n = np.array([nx, ny, nz])
        n = n / np.linalg.norm(n)
        
    b = Bloch()
    b.alpha = alpha
    b.n = n
    b.theta = theta
    return b


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
# Calculate cot(pi/8)
cot_pi_8 = 1.0 / np.tan(np.pi / 8.0)

n1_unnorm = np.array([-cot_pi_8, 1.0, cot_pi_8])
n2_unnorm = np.array([1.0/np.sqrt(2), np.sqrt(2)*cot_pi_8, -1.0/np.sqrt(2)])

n1 = n1_unnorm / np.linalg.norm(n1_unnorm)
n2 = n2_unnorm / np.linalg.norm(n2_unnorm)

a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)

def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)
    """
    phi = b.theta / 2.0
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    
    s1 = -np.dot(b.n, a1) * s_phi 
    s2 = -np.dot(b.n, a2) * s_phi
    s3 = np.dot(b.n, a3) * s_phi
    
    gamma_plus_alpha = np.arctan2(s1, c_phi)
    
    gamma_minus_alpha = np.arctan2(s3, s2)
    
    hypot_s2_s3 = np.hypot(s2, s3)
    hypot_c_s1 = np.hypot(c_phi, s1)
    beta = np.arctan2(hypot_s2_s3, hypot_c_s1)
    
    alpha = (gamma_plus_alpha - gamma_minus_alpha) / 2.0
    gamma = (gamma_plus_alpha + gamma_minus_alpha) / 2.0
    
    alpha = alpha % TWO_PI
    beta = beta % TWO_PI
    gamma = gamma % TWO_PI
    
    return alpha, beta, gamma, b.alpha

def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    """
    target = angle % TWO_PI
    
    k = 1
    while True:
        current = (k * LAMBDA_PI) % TWO_PI
        
        dist = min(abs(current - target), TWO_PI - abs(current - target))
        
        if dist <= tolerance:
            return k
            
        k += 1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:
        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)
    """
    b = to_bloch(u)
    alpha, beta, gamma, _global_phase = n1n2n1_angles(b)
    
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)
    
    return k, l, m

#test_unitary = H 
#
## 2. Test to_bloch
#print("Testing to_bloch...")
#bloch_obj = to_bloch(test_unitary)
#print(f"Alpha: {bloch_obj.alpha}, Theta: {bloch_obj.theta}, Axis: {bloch_obj.n}")
#
## 3. Test n1n2n1_angles
#print("\nTesting n1n2n1_angles...")
#angles = n1n2n1_angles(bloch_obj)
#print(f"Angles (alpha, beta, gamma, global_phase): {angles}")
#
## 4. Test full decomposition
#print("\nTesting decompose_2x2...")
## Using a larger tolerance for a quicker test
#powers = decompose_2x2(test_unitary, tolerance=0.1) 
#print(f"M1, M2, M1 powers (k, l, m): {powers}")