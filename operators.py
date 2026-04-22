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
def ContinuousCrossoverCauchy(x1, x2): #x1 and x2 are males and females parents respectively
    # Generate random value within specific range

    # Cauchy noise 
    u1 = np.random.uniform(0.0, 1.0, size=(x1.shape[0], x1.shape[1]))
    u2 = np.random.uniform(0.0, 1.0, size=(x2.shape[0], x2.shape[1]))
    noise1 = np.tan(np.pi * (u1 - 0.5))
    noise2 = np.tan(np.pi * (u2 - 0.5))
    y1 = x1 + x1 * noise1
    y2 = x2 + x2 * noise2
    
    return y1, y2
# %%
def ContinuousCrossoverGaussian(x1, x2, sigma=0.1):
    
    # Gaussian noise: N(0,1) * sigma.
    noise1 = np.random.normal(0.0, 1.0, size=(x1.shape[0], x1.shape[1])) * sigma
    noise2 = np.random.normal(0.0, 1.0, size=(x2.shape[0], x2.shape[1])) * sigma
    y1 = x1 + x1 * noise1
    y2 = x2 + x2 * noise2
    
    return y1, y2
# %%
# Mutate the genes of offspring
# Adds a random mutation to portion of population in otder to explore new areas of search space 
def ContinuousMutation(x1, problem):
    x1 = x1 + (problem['VarMax'] - problem['VarMin']) * np.random.uniform(low = -1, high = 1, size = (x1.shape[0], x1.shape[1]))
    return x1