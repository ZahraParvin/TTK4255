import numpy as np

def triangulate_many(u1, u2, P1, P2):
    """
    Arguments
        u: Image coordinates in image 1 and 2
        P: Projection matrix K[R t] for image 1 and 2
    Returns
        X: Homogeneous coordinates of 3D points (shape 4 x n)
    """
    n = u1.shape[1]
    X = np.zeros((4, n)) 
    
    for i in range(n):
        A = np.vstack([
            u1[0, i]*P1[2,:] - P1[0,:],
            u1[1, i]*P1[2,:] - P1[1,:],
            u2[0, i]*P2[2,:] - P2[0,:],
            u2[1, i]*P2[2,:] - P2[1,:]
        ])
        _, _, Vt = np.linalg.svd(A)
        X[:, i] = Vt[-1]  
        X[:, i] /= X[3, i]
        
    return X
