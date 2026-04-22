def printingperiter(problem, it, gbest, namemethod, funcevals, curtrial, progress = None, elapsed_str = None, eta_str = None):
    if progress is None:
        progress_str = "unknown" 
    else:
        progress_str = f"{100.0 * progress:.2f}%"
    
    if elapsed_str is None:
        elapsed_str = "unknown"
        
    if eta_str is None:
        eta_str = "unknown"
    
    print(
        'Problem: {}, Method: {}, Trial: {}, Iteration: {}, Function Evaluations: {}, Best Cost = {}'.format(
            problem['CostFunction'].__name__,
            namemethod,
            curtrial + 1,
            it,
            funcevals,
            gbest['cost']
        ),
        flush=True,
    ) 
    return