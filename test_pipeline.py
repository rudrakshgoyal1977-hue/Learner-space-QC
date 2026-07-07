import numpy as np
import matplotlib.pyplot as plt
from decompose import decompose_to_ht, circuit_to_unitary, error_up_to_phase

def generate_random_2qubit_unitary():
    """Generates a random 4x4 unitary matrix using Haar measure (via QR)."""
    X = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q @ np.diag(ph)

def check_gate_scaling():
    # Generate a fixed random 4x4 unitary matrix
    u = generate_random_2qubit_unitary()
    
    tolerances = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    gate_counts = []
    
    print(f"{'Tolerance (Error)':<18} | {'Total Gates':<12} | {'Reconstruction Error':<20}")
    print("-" * 58)
    
    for tol in tolerances:
        circuit = decompose_to_ht(u, error=tol)
        num_gates = len(circuit)
        gate_counts.append(num_gates)
        
        rebuilt_u = circuit_to_unitary(circuit)
        actual_err = error_up_to_phase(u, rebuilt_u)
        
        print(f"{tol:<18} | {num_gates:<12} | {actual_err:<20.2e}")
        
    # Plot the scaling behavior on a log-log scale
    plt.figure(figsize=(8, 5))
    plt.plot(tolerances, gate_counts, marker='o', linestyle='-', color='purple', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Error Tolerance ($\epsilon$) [Log Scale]')
    plt.ylabel('Number of Gates [Log Scale]')
    plt.title('Gate Count Scaling vs. Error Tolerance for a 4x4 Unitary')
    plt.gca().invert_xaxis()  # Smaller error tolerance moves to the right
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    check_gate_scaling()