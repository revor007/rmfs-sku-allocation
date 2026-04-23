# %%
# Import Library
import numpy as np
# %%
def ContinuousCrossover(x1, x2, gamma):
    
    alpha = np.random.uniform(
        -gamma,
        1 + gamma,
        size = (x1.shape[0], x1.shape[1])
    )
    
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    
    return y1, y2
# %%
def ContinuousCrossoverCauchy(x1, x2, rr=None, nvariation=None, NP=None): #x1 and x2 are males and females parents respectively
    # Apply Cauchy variation only on nvariation randomly selected dimensions.
    total_dims = int(x1.size)
    if total_dims == 0:
        return x1, x2

    if rr is None:
        rr = float(np.random.rand())
    eps_rr = np.finfo(np.float32).eps
    rr = float(np.clip(rr, eps_rr, 1.0 - eps_rr))

    if nvariation is None:
        if NP is None:
            NP = total_dims
        nvariation = int(float(NP) * rr)
    nvariation = int(np.clip(nvariation, 0, total_dims))

    y1 = np.asarray(x1, dtype=np.float32).copy()
    y2 = np.asarray(x2, dtype=np.float32).copy()
    if nvariation == 0:
        return y1, y2

    flat1 = y1.reshape(-1)
    flat2 = y2.reshape(-1)

    # Select independently per offspring.
    idx1 = np.random.choice(total_dims, size=nvariation, replace=False)
    idx2 = np.random.choice(total_dims, size=nvariation, replace=False)

    # Cauchy noise for selected dimensions only.
    u1 = np.random.uniform(0.0, 1.0, size=nvariation)
    u2 = np.random.uniform(0.0, 1.0, size=nvariation)
    noise1 = np.tan(np.pi * (u1 - 0.5))
    noise2 = np.tan(np.pi * (u2 - 0.5))

    flat1[idx1] = flat1[idx1] + flat1[idx1] * noise1
    flat2[idx2] = flat2[idx2] + flat2[idx2] * noise2
    
    return y1, y2
# %%
def ContinuousCrossoverGaussian(x1, x2, sigma=0.1, rr=None, nvariation=None, NP=None):
    # Apply Gaussian variation only on nvariation randomly selected dimensions.
    total_dims = int(x1.size)
    if total_dims == 0:
        return x1, x2

    if rr is None:
        rr = float(np.random.rand())
    eps_rr = np.finfo(np.float32).eps
    rr = float(np.clip(rr, eps_rr, 1.0 - eps_rr))

    if nvariation is None:
        if NP is None:
            NP = total_dims
        nvariation = int(float(NP) * rr)
    nvariation = int(np.clip(nvariation, 0, total_dims))

    y1 = np.asarray(x1, dtype=np.float32).copy()
    y2 = np.asarray(x2, dtype=np.float32).copy()
    if nvariation == 0:
        return y1, y2

    flat1 = y1.reshape(-1)
    flat2 = y2.reshape(-1)

    # Select independently per offspring.
    idx1 = np.random.choice(total_dims, size=nvariation, replace=False)
    idx2 = np.random.choice(total_dims, size=nvariation, replace=False)

    # Gaussian noise for selected dimensions only.
    noise1 = np.random.normal(0.0, 1.0, size=nvariation) * sigma
    noise2 = np.random.normal(0.0, 1.0, size=nvariation) * sigma

    flat1[idx1] = flat1[idx1] + flat1[idx1] * noise1
    flat2[idx2] = flat2[idx2] + flat2[idx2] * noise2
    
    return y1, y2
# %%
# Mutate the genes of offspring
# Adds a random mutation to portion of population in otder to explore new areas of search space 
def ContinuousMutation(x1, problem):
    x1 = x1 + (problem['VarMax'] - problem['VarMin']) * np.random.uniform(low = -1, high = 1, size = (x1.shape[0], x1.shape[1]))
    return x1