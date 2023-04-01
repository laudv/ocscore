import veritas
import time
from veritas import KantchelianAttack
from util import *

def get_closest_adversarial_examples(d, indices, at, N, start_delta, max_time=10):
    kan_advs = []
    nfail = 0

    if N > len(indices):
        print(f"WARNING: reducing N from {N} to {len(indices)}")
        N = len(indices)

    for i in indices:
        print(f"CLOSEST {i}, {len(kan_advs)}/{N} (nfail={nfail})")
        example = d.X.iloc[i, :].to_numpy()
        label = int(d.y[i])
        target_output = int(not bool(label))

        tstart = time.time()
        #kan = KantchelianAttack(at, target_output=target_output,
        #                        example=example, silent=True, guard=1e-5)
        #kan.optimize()
        #if not kan.has_solution():
        #    continue
        #adv_example, adv_output = kan.solution()[:2]
        #print("KAN ADV OUTPUT ", adv_output, "expected", target_output)

        at0, at1 = (at, None) if label else (None, at) # minimize if y==1, max if y==0
        rob = veritas.VeritasRobustnessSearch(at0, at1,
                example=example, mem_capacity=16*1024*1024*1024,
                start_delta=start_delta, max_time=max_time)
        rob.search()
        if len(rob.generated_examples) > 0:
            adv_example = rob.generated_examples[-1]
        else:
            print(f"i={i}: NO ADV.EX. FOUND, y={at.eval(example)[0]:.3f}")
            nfail += 1
            continue

        tstop = time.time()
        #analyze_advs_boxes(d, at, [{"base_example": example, "adv_example": adv_example}])
        verify_example_outcome(at, adv_example, target_output,
                               f"KAN adv example {i} label doesn't match")

        kan_advs.append({
            "index": i,
            "time": tstop - tstart,
            "adv_example": adv_example,
            "base_example": example,
            "base_label": label,
            "adv_score": at.eval(adv_example)[0],
            "linf": linf(example, adv_example)
        })

        if len(kan_advs) >= N: break
        print()

    return kan_advs

def get_veritas_search(at, example, delta):
    ver = veritas.Search.max_output(at)
    ver.set_mem_capacity(16*1024*1024*1024)
    ver.prune([veritas.Domain(x-delta, x+delta) for x in example])
    ver.stop_when_upper_less_than = 0.0
    ver.auto_eps = False
    ver.eps = 0.05
    ver.stop_when_optimal = True
    ver.stop_when_num_solutions_exceeds = 1
    ver.reject_solution_when_output_less_than = 0.0
    ver.max_focal_size = 10000

    return ver

def get_adversarial_examples(d, indices, delta, at, N):
    ver_advs = []
    max_time = 30
    failcount = 0

    if N > len(indices):
        print(f"WARNING: reducing N from {N} to {len(indices)}")
        N = len(indices)

    for i in indices:
        example = d.X.iloc[i, :].to_numpy()
        label = int(d.y[i])
        target_output = int(not bool(label))

        at_ver = at if target_output else at.negate_leaf_values()

        tstart = time.time()
        current_delta = delta
        ver = get_veritas_search(at_ver, example, current_delta)
    
        adv_example = None
        while True:

            try:
                ver.step_for(0.25, 250)
            except RuntimeError as e:
                print("Out of memory")
                print(e)
                break
            upper_bound = ver.current_bounds()[1]

            if ver.num_solutions() > 0 and upper_bound >= 0.0:
                sol = ver.get_solution(0)
                adv_example = veritas.get_closest_example(sol, example)
                print(f"VERITAS {i} solution found in {time.time() - tstart:.2f}s")
                break
            elif time.time() - tstart > max_time:
                break
            elif upper_bound < 0.0:
                print(f"VERITAS {i} increasing delta")
                current_delta *= 2.0
                ver = get_veritas_search(at_ver, example, current_delta)
        tstop = time.time()

        if adv_example is None:
            failcount += 1
            print(f"No adversarial example found... count={failcount}")
            continue

        assert adv_example is not None
        ver_advs.append({
            "index": i,
            "time": tstop - tstart,
            "adv_example": adv_example,
            "base_example": example,
            "base_label": label,
            "adv_score": at.eval(adv_example)[0],
            "delta": current_delta,
            "linf": linf(example, adv_example)
        })

        if len(ver_advs) >= N: break
        


    return ver_advs
