import numpy as np

def estimate_E(B1, B2):
    # B1, B2: Arrays of size 3 x n containing back-projection vectors in image 1 and 2, respectively.
    n = B1.shape[1]
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = np.kron(B1[:, i], B2[:, i]).reshape(-1)
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    
    # Enforcing the rank-2 constraint
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0  # Setting the smallest singular value to zero
    E = U @ np.diag(S) @ Vt
    
    return E
