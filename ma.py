# Project Title: Mayfly optimization algorithm (MA) for RMFS
#
# Developers: K. Zervoudakis & S. Tsafarakis, modified by Lukman Al Rasyid
#
# Contact Info: lukmanalrasyid63@gmail.com
#               Engineering Management,
#               Institut Teknologi Bandung, Indonesia

# %%
# Library imports
import numpy as np
import random
from operators import ContinuousCrossover, ContinuousCrossoverCauchy, ContinuousCrossoverGaussian, ContinuousMutation as ContinousMutation
from printingIter import printingperiter

# %%

# Mayfly Algorithm
def MA(problem, IterPrint, MaxIter, MaxFuncEvals, curtrial, initialpop, mPopSize, fPopSize,
       a1=1.0, a2=1.5, a3=2.0, beta=2, dance=5, fl=1, dance_damp=0.8, fl_damp=0.99,
       nc=20, gmax=0.8, gmin=0.8, gamma=0.4, on_iter = None, on_position = None):
    namemethod = 'MA'

    # Initial population size guard
    required_init_size = mPopSize + fPopSize
    if len(initialpop) < required_init_size:
        raise ValueError(
            f"initialpop has {len(initialpop)} candidates, but MA needs at least {required_init_size} (mPopSize+fPopSize)."
        )

    # Variable bounds for position
    VarMin = np.asarray(problem['VarMin'], dtype=np.float32)
    VarMax = np.asarray(problem['VarMax'], dtype=np.float32)

    # Sets mutant counts: at least 1 mutant
    nmf = max(1, round(0.05 * fPopSize))

    # Define velocity limits for position updates
    VelMax = (0.1 * (VarMax - VarMin)).astype(np.float32, copy=False)
    VelMin = (-VelMax).astype(np.float32, copy=False)

    # Extract RMFS problem
    RepairFunction = problem['RepairFunction']
    CostFunction = problem['CostFunction']

    nVar = problem['nVar']
    shape = (nVar[0], nVar[1])
    NP = int(mPopSize + fPopSize)

    # Gravity coefficient
    g = float(gmax)

    # Function evaluation counter
    funcevals = -1

    # Convergence history
    convergence = []

    # Empty particle templates
    def empty_male_particle():
        particle = problem['ProblemStructure'].copy()
        particle.update({
            'velocity': None,
            'best_position': None,
            'best_cost': None,
        })
        return particle

    def empty_female_particle():
        particle = problem['ProblemStructure'].copy()
        particle.update({
            'velocity': None,
        })
        return particle

    # Initialize global best
    gbest = problem['ProblemStructure'].copy()
    gbest['cost'] = -np.inf

    def update_global_best(candidate):
        if candidate['cost'] > gbest['cost']:
            gbest['position'] = candidate['position'].copy()
            if 'position_bin' in candidate and candidate['position_bin'] is not None:
                gbest['position_bin'] = candidate['position_bin'].copy()
            gbest['quantity_num'] = candidate['quantity_num'].copy()
            gbest['compartment_num'] = candidate['compartment_num'].copy()
            gbest['cost'] = float(candidate['cost'])
            if 'ProblemPosition' in candidate:
                gbest['ProblemPosition'] = candidate['ProblemPosition'].copy()

    def function_budget_reached():
        return (
            problem['StopCriterion'] == 'Function Evaluations'
            and funcevals >= MaxFuncEvals
        )

    def evaluate_candidate(candidate):
        nonlocal funcevals
        candidate = RepairFunction(candidate, problem)
        candidate = CostFunction(candidate, problem)
        funcevals += 1
        update_global_best(candidate)
        return candidate

    # Create initial male population
    pop = []
    for i in range(mPopSize):
        cand = empty_male_particle()
        cand['position'] = np.asarray(initialpop[i]['position'], dtype=np.float32).copy()
        cand['quantity_num'] = np.asarray(initialpop[i]['quantity_num'], dtype=np.int32).copy()
        cand['compartment_num'] = np.asarray(initialpop[i]['compartment_num'], dtype=np.int32).copy()
        cand['velocity'] = np.zeros(shape, dtype=np.float32)
        cand['cost'] = float(initialpop[i]['cost'])
        cand['best_position'] = cand['position'].copy()
        cand['best_cost'] = cand['cost']
        pop.append(cand)
        update_global_best(cand)
        funcevals += 1

    # Create initial female population
    popf = []
    for i in range(fPopSize):
        cand = empty_female_particle()
        src = initialpop[mPopSize + i]
        cand['position'] = np.asarray(src['position'], dtype=np.float32).copy()
        cand['quantity_num'] = np.asarray(src['quantity_num'], dtype=np.int32).copy()
        cand['compartment_num'] = np.asarray(src['compartment_num'], dtype=np.int32).copy()
        cand['velocity'] = np.zeros(shape, dtype=np.float32)
        cand['cost'] = float(src['cost'])
        popf.append(cand)
        update_global_best(cand)
        funcevals += 1

    # Store initial best once
    convergence.append(gbest['cost'])

    # Main loop of MA
    it = 0
    while (
        (problem['StopCriterion'] == 'Iterations' and it < MaxIter)
        or (problem['StopCriterion'] == 'Function Evaluations' and funcevals < MaxFuncEvals)
    ):
        # Update Females
        for i in range(fPopSize):
            if i < mPopSize and popf[i]['cost'] < gbest['cost']:
                rmf = np.abs(pop[i]['position'] - popf[i]['position'], dtype=np.float32)
                attract = a3 * np.exp(-beta * (rmf ** 2), dtype=np.float32) * (pop[i]['position'] - popf[i]['position'])
                popf[i]['velocity'] = g * popf[i]['velocity'] + attract
            else:
                rnd = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32, copy=False)
                popf[i]['velocity'] = g * popf[i]['velocity'] + np.float32(fl) * rnd

            np.clip(popf[i]['velocity'], VelMin, VelMax, out=popf[i]['velocity'])
            popf[i]['position'] += popf[i]['velocity']
            np.clip(popf[i]['position'], VarMin, VarMax, out=popf[i]['position'])

            popf[i] = evaluate_candidate(popf[i])
            if function_budget_reached():
                break
        if function_budget_reached():
            break

        # Update Males
        for i in range(mPopSize):
            if pop[i]['cost'] < gbest['cost']:
                rpbest = np.abs(pop[i]['best_position'] - pop[i]['position'], dtype=np.float32)
                rgbest = np.abs(gbest['position'] - pop[i]['position'], dtype=np.float32)
                term_pb = a1 * np.exp(-beta * (rpbest ** 2), dtype=np.float32) * (pop[i]['best_position'] - pop[i]['position'])
                term_gb = a2 * np.exp(-beta * (rgbest ** 2), dtype=np.float32) * (gbest['position'] - pop[i]['position'])
                pop[i]['velocity'] = g * pop[i]['velocity'] + term_pb + term_gb
            else:
                rnd = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32, copy=False)
                pop[i]['velocity'] = g * pop[i]['velocity'] + np.float32(fl) * rnd

            np.clip(pop[i]['velocity'], VelMin, VelMax, out=pop[i]['velocity'])
            pop[i]['position'] += pop[i]['velocity']
            np.clip(pop[i]['position'], VarMin, VarMax, out=pop[i]['position'])

            pop[i] = evaluate_candidate(pop[i])

            # Update each male's personal best
            if pop[i]['cost'] > pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost']

            if function_budget_reached():
                break
        if function_budget_reached():
            break

        # Sort mayflies
        pop.sort(key=lambda x: x['cost'], reverse=True)
        popf.sort(key=lambda x: x['cost'], reverse=True)

        # Mate the mayflies
        pair_count = min(int(nc / 2), len(pop), len(popf))
        for i in range(pair_count):
            # Adapted crossover: uses Cauchy or Gaussian noise based on the current progress of the algorithm, which helps maintain a balance between exploration and exploitation as the search progresses
            male_child_adapt = empty_male_particle()
            female_child_adapt = empty_female_particle()

            male_child_base = empty_male_particle()
            female_child_base = empty_female_particle()

            # Base crossover: always uses the same method (without noise) to ensure that a stable, exploitation-focused option is available in the selection process, which helps maintain a balance between exploration and exploitation
            child_m, child_f = ContinuousCrossover(pop[i]['position'], popf[i]['position'], gamma)    
            male_child_base['position'] = np.asarray(child_m, dtype=np.float32)
            female_child_base['position'] = np.asarray(child_f, dtype=np.float32)
            np.clip(male_child_base['position'], VarMin, VarMax, out=male_child_base['position'])
            np.clip(female_child_base['position'], VarMin, VarMax, out=female_child_base['position'])

            male_child_base['quantity_num'] = np.zeros(shape, dtype=np.int32)
            female_child_base['quantity_num'] = np.zeros(shape, dtype=np.int32)
            male_child_base['compartment_num'] = np.zeros(shape, dtype=np.int32)
            female_child_base['compartment_num'] = np.zeros(shape, dtype=np.int32)

            male_child_base['velocity'] = np.zeros(shape, dtype=np.float32)
            female_child_base['velocity'] = np.zeros(shape, dtype=np.float32)
            male_child_base['best_position'] = male_child_base['position'].copy()

            # Define o using current iteration t and total iterations NG.
            if MaxIter > 0:
                t = float(min(it + 1, MaxIter))
                NG = float(MaxIter)
                o = np.exp(-((1.0 - (t / NG)) ** (1.0 / np.e)))
            else:
                o = 1.0

            # Rate-based sparse variation: rr in (0,1), nvariation = NP * rr.
            eps_rr = np.finfo(np.float32).eps
            rr = float(np.clip(np.random.rand(), eps_rr, 1.0 - eps_rr))
            nvariation = int(NP * rr)
            
            # If o < 1/2 (early stage), use Cauchy noise for more exploration; else (late stage), use Gaussian noise for more exploitation
            if o < (1/2):
                child_m, child_f = ContinuousCrossoverCauchy(
                    male_child_base['position'],
                    female_child_base['position'],
                    rr=rr,
                    nvariation=nvariation,
                    NP=NP,
                )
            else:
                child_m, child_f = ContinuousCrossoverGaussian(
                    male_child_base['position'],
                    female_child_base['position'],
                    sigma=0.1,
                    rr=rr,
                    nvariation=nvariation,
                    NP=NP,
                )

            male_child_adapt['position'] = np.asarray(child_m, dtype=np.float32)
            female_child_adapt['position'] = np.asarray(child_f, dtype=np.float32)
            np.clip(male_child_adapt['position'], VarMin, VarMax, out=male_child_adapt['position'])
            np.clip(female_child_adapt['position'], VarMin, VarMax, out=female_child_adapt['position'])

            male_child_adapt['quantity_num'] = np.zeros(shape, dtype=np.int32)
            female_child_adapt['quantity_num'] = np.zeros(shape, dtype=np.int32)
            male_child_adapt['compartment_num'] = np.zeros(shape, dtype=np.int32)
            female_child_adapt['compartment_num'] = np.zeros(shape, dtype=np.int32)

            male_child_adapt['velocity'] = np.zeros(shape, dtype=np.float32)
            female_child_adapt['velocity'] = np.zeros(shape, dtype=np.float32)
            male_child_adapt['best_position'] = male_child_adapt['position'].copy()
            

            male_child_adapt = evaluate_candidate(male_child_adapt)
            if function_budget_reached():
                break
            female_child_adapt = evaluate_candidate(female_child_adapt)
            if function_budget_reached():
                break
            male_child_base = evaluate_candidate(male_child_base)
            if function_budget_reached():
                break
            female_child_base = evaluate_candidate(female_child_base)
            if function_budget_reached():
                break

            male_child_adapt['best_cost'] = male_child_adapt['cost']
            male_child_base['best_cost'] = male_child_base['cost']

            # Greedy selection: only the better child (adapted or base) is added to the population, which helps maintain a balance between exploration and exploitation
            if male_child_adapt['cost'] > male_child_base['cost']:
                pop.append(male_child_adapt)
            else:
                pop.append(male_child_base)
                
            if female_child_adapt['cost'] > female_child_base['cost']:
                popf.append(female_child_adapt)
            else:
                popf.append(female_child_base)

            if function_budget_reached():
                break
        if function_budget_reached():
            break

        # Mutation of the mayflies
        for _ in range(nmf):
            a = random.randint(0, fPopSize - 1)
            mutant = empty_female_particle()
            mutant['position'] = np.asarray(ContinousMutation(popf[a]['position'], problem), dtype=np.float32)
            np.clip(mutant['position'], VarMin, VarMax, out=mutant['position'])
            mutant['quantity_num'] = np.zeros(shape, dtype=np.int32)
            mutant['compartment_num'] = np.zeros(shape, dtype=np.int32)
            mutant['velocity'] = np.zeros(shape, dtype=np.float32)

            mutant = evaluate_candidate(mutant)
            popf.append(mutant)
            if function_budget_reached():
                break
        if function_budget_reached():
            break

        # Sort and truncate
        pop.sort(key=lambda x: x['cost'], reverse=True)
        del pop[mPopSize:]
        popf.sort(key=lambda x: x['cost'], reverse=True)
        del popf[fPopSize:]

        # Gravity coefficient update
        if MaxIter > 0:
            g = float(gmax - ((gmax - gmin) / MaxIter) * it)

        # Reduction of random walk coefficient
        dance *= dance_damp
        fl *= fl_damp

        # Store one convergence point per iteration
        convergence.append(gbest['cost'])

        if it % IterPrint == 0:
            printingperiter(problem, it, gbest, namemethod, funcevals, curtrial)

        if on_iter is not None:
            on_iter(it, gbest)
        
        if on_position is not None:
            all_agents = pop + popf
            coords = np.empty((len(all_agents), 2), dtype=np.float32)
            
            for k, agent in enumerate(all_agents):
                flat = np.ravel(np.asarray(agent['position'], dtype=np.float32))
                coords[k, 0] = flat[0] if flat.size > 0 else 0.0
                coords[k, 1] = flat[1] if flat.size > 1 else 0.0
            
            on_position(it, coords)
        
        it += 1

    # Persist the final best solution so the caller can export X, Q, and Y
    best_position = gbest['position'].copy()
    best_position_bin = gbest.get('position_bin', None)
    if best_position_bin is None:
        best_position_bin = np.clip(np.round(best_position), 0, 1).astype(np.float32)
    else:
        best_position_bin = best_position_bin.copy()

    best_quantity_num = gbest.get('quantity_num', None)
    best_compartment_num = gbest.get('compartment_num', None)

    if best_quantity_num is None or best_compartment_num is None:
        tmp_best = problem['ProblemStructure'].copy()
        tmp_best['position'] = best_position.copy()
        tmp_best['quantity_num'] = np.zeros(shape, dtype=np.int32)
        tmp_best['compartment_num'] = np.zeros(shape, dtype=np.int32)
        tmp_best = RepairFunction(tmp_best, problem)

        best_quantity_num = tmp_best['quantity_num'].copy()
        best_compartment_num = tmp_best['compartment_num'].copy()
    else:
        best_quantity_num = best_quantity_num.copy()
        best_compartment_num = best_compartment_num.copy()

    problem['LastBest'] = {
        'position': best_position_bin,
        'position_cont': best_position,
        'quantity_num': best_quantity_num,
        'compartment_num': best_compartment_num,
        'cost': gbest['cost'],
    }

    if 'ProblemPosition' in gbest:
        results = np.array(
            [namemethod, gbest['ProblemPosition'], gbest['cost'], convergence, best_position_bin],
            dtype=object,
        )
    else:
        results = np.array([namemethod, best_position_bin, gbest['cost'], convergence], dtype=object)
    
    return results