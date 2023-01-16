import os

from util import *

import lt_attack
import veritas_attack
import cube_attack

def get_adversarial_fname(fname, replace):
    return fname.replace("<REPLACE>", replace)

def get_adversarial_fnames(fname):
    return [get_adversarial_fname(fname, x) for x in ["advs_lt", "advs_kan",
                                                      "advs_ver", "advs_cub"]]

def get_adversarial_examples(fname, d, at, indices, N, cache=True):
    nfolds = d.nfolds

    gen = np.random.default_rng(d.seed)
    lt_fname, kan_fname, ver_fname, cub_fname = get_adversarial_fnames(fname)

    # [1] LT ATTACK (Zhang et al. [2020a])
    gen.shuffle(indices)
    if os.path.exists(lt_fname):
        lt_advs = load(lt_fname)
    else:
        lt_advs = lt_attack.get_adversarial_examples(d, indices, at, N, debug=False)
        if cache: dump(lt_fname, lt_advs)

    # [2] CLOSEST ADVERSARIAL EXAMPLES (Kantchelian et al. [2016], but we use
    # Veritas to compute them)
    gen.shuffle(indices)
    if os.path.exists(kan_fname):
        kan_advs = load(kan_fname)
    else:
        kan_advs = veritas_attack.get_closest_adversarial_examples(
                d, indices, at, N, max_time=MAX_TIME[d.name()])
        if cache: dump(kan_fname, kan_advs)

    # Use a delta/eps value relative to the linfs found for the closest
    # adversarial examples for Veritas and cube
    closest_linfs_dict = {x["index"] : x["linf"] for x in kan_advs}
    reference_linf = np.quantile(list(closest_linfs_dict.values()), 0.8)

    print("Reference L-inf is", reference_linf)

    # [3] VERITAS
    gen.shuffle(indices)
    if os.path.exists(ver_fname):
        ver_advs = load(ver_fname)
    else:
        ver_advs = veritas_attack.get_adversarial_examples(
                d, indices, reference_linf, at, N)
        if cache: dump(ver_fname, ver_advs)

    # [4] CUBE ATTACK, slightly modified version
    gen.shuffle(indices)
    if os.path.exists(cub_fname):
        cub_advs = load(cub_fname)
    else:
        cub_advs = cube_attack.get_adversarial_examples(
                d, indices, reference_linf*5.0, at, N)
        if cache: dump(cub_fname, cub_advs)

    return lt_advs, kan_advs, ver_advs, cub_advs
