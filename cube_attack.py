import time
from util import *


class _Fun:
    def __init__(self, at):
        self.at = at

    # ADAPTED FROM: https://github.com/max-andr/provably-robust-boosting/blob/577f25dce2a28c0503f0f7c69c49dc413e891260/classifiers.py#L35
    def fmargin(self, X, y):
        return (2.0 * y - 1.0) * self.at.eval(X)

# ADAPTED FROM: https://github.com/max-andr/provably-robust-boosting/blob/577f25dce2a28c0503f0f7c69c49dc413e891260/attacks.py#L25
def cube_attack(rng, f, X, y, eps, n_trials, p=0.5, deltas_init=None,
                independent_delta=False, min_val=0.0, max_val=1.0):
    """
    A simple, but efficient black-box attack that just adds random steps of
    values in {-2eps, 0, 2eps} (i.e., the considered points are always corners).
    The random change is added if the loss decreases for a particular point.
    The only disadvantage of this method is that it will never find decision
    regions inside the Linf-ball which do not intersect any corner. But tight
    LRTE (compared to RTE/URTE) suggest that this doesn't happen.

        `f` is any function that has f.fmargin() method that returns class scores.
        `eps` can be a scalar or a vector of size X.shape[0].
        `min_val`, `max_val` are min/max allowed values for values in X (e.g. 0
         and 1 for images). This can be adjusted depending on the feature range
         of the data. It's also possible to specify the as numpy vectors.
    """
    assert type(eps) is float or type(eps) is np.ndarray

    p_neg_eps = p/2  # probability of sampling -2eps
    p_pos_eps = p/2  # probability of sampling +2eps
    p_zero = 1 - p  # probability of not doing an update
    num, dim = X.shape
    # independent deltas work better for adv. training but slow down attacks
    size_delta = (num, dim) if independent_delta else (1, dim)

    if deltas_init is None:
        deltas_init = np.zeros(size_delta)
    # this init is important, s.t. there is no violation of bounds
    f_x_vals_min = f.fmargin(X, y)

    if deltas_init is not None:  # evaluate the provided deltas and take them if they are better
        X_adv = np.clip(X + deltas_init, np.maximum(min_val, X - eps), np.minimum(max_val, X + eps))
        deltas = X_adv - X  # because of the projection above, the new delta vector is not just +-eps
        f_x_vals = f.fmargin(X_adv, y)
        idx_improved = f_x_vals < f_x_vals_min
        f_x_vals_min = idx_improved * f_x_vals + ~idx_improved * f_x_vals_min
        deltas = idx_improved[:, None] * deltas_init + ~idx_improved[:, None] * deltas
    else:
        deltas = deltas_init

    i_trial = 0
    while i_trial < n_trials:
        # +-2*eps is *very* important to escape local minima; +-eps has very unstable performance
        new_deltas = rng.choice([-2*eps, 0, 2*eps], p=[p_neg_eps, p_zero, p_pos_eps], size=size_delta)
        #new_deltas = 2 * eps * new_deltas  # if eps is a vector, then it's an outer product num x 1 times 1 x dim
        X_adv = np.clip(X + deltas + new_deltas, np.maximum(min_val, X - eps), np.minimum(max_val, X + eps))
        new_deltas = X_adv - X  # because of the projection above, the new delta vector is not just +-eps
        f_x_vals = f.fmargin(X_adv, y)
        idx_improved = f_x_vals < f_x_vals_min

        #f_x_vals_min = idx_improved * f_x_vals + ~idx_improved * f_x_vals_min
        #deltas = idx_improved[:, None] * new_deltas + ~idx_improved[:, None] * deltas

        f_x_vals_min = np.where(idx_improved, f_x_vals, f_x_vals_min)
        deltas = np.where(idx_improved[:, None], new_deltas, deltas)

        i_trial += 1

    return X_adv

# pertrub single attribute per trial
def cube_attack2(rng, f, X, y, eps, ntrials, attributes, p=0.5, deltas_init=None,
                independent_delta=False, min_val=0.0, max_val=1.0):

    xadv = X.copy()
    xadv_tmp = X.copy()
    #done = np.zeros(X.shape[0], dtype=bool)
    fmargin_min = f.fmargin(xadv, y)
    xlo, xhi = np.maximum(X-eps, min_val), np.minimum(max_val, X+eps)
    diff = np.zeros(X.shape)
    for trial in range(ntrials):
        diff.fill(0.0)
        #sel = rng.integers(0, X.shape[1], (X.shape[0], 1))
        sel = rng.choice(attributes, size=(X.shape[0], 1))
        dir = eps if trial//2==0 else -eps
        np.put_along_axis(diff, sel, dir, axis=1)
        diff *= rng.random((X.shape[0], 1)) # magnitude
        np.clip(xadv+diff, xlo, xhi, out=xadv_tmp)
        fmargin = f.fmargin(xadv_tmp, y)
        #idx_improved = (fmargin < fmargin_min) & ~done
        idx_improved = (fmargin < fmargin_min)
        #done |= fmargin < 0.0
        #print("imp", idx_improved)
        #print("don", done)
        #print("pre", fmargin_min)
        #print("now", fmargin)
        fmargin_min = np.where(idx_improved, fmargin, fmargin_min)
        xadv = np.where(idx_improved[:, np.newaxis], xadv_tmp, xadv)
    return xadv

def get_adversarial_examples(d, indices, eps, at, N, ntrials=1000, seed=1):
    f = _Fun(at)
    attributes = np.array(list(at.get_splits().keys()))
    #print("attributes", attributes)
    rng = np.random.default_rng(seed)

    if N > len(indices):
        print(f"WARNING: reducing N from {N} to {len(indices)}")
        N = len(indices)

    #eps = np.array(closest_linfs).reshape((len(indices), 1)) * delta_multiplier
    chunk_size = 100
    advs = []
    i = 0
    dur = 0.0
    fail_count = 0

    while len(advs) < N and i < len(indices):
        js = range(i, min(len(indices), i+chunk_size))
        r = indices[js]
        #print("CHUNK", r, js)
        Xsub = d.X.iloc[r, :].to_numpy()
        ysub = d.y[r].to_numpy()
        #epssub = eps[js, :]

        tstart = time.time()
        adv_examples = cube_attack2(rng, f, Xsub, ysub, eps, ntrials, attributes, p=0.1)
        dur += time.time() - tstart
        offset = len(advs)

        # remove fails
        for j in range(adv_examples.shape[0]):
            adv_example = adv_examples[j, :]
            example = d.X.iloc[indices[i], :].to_numpy()
            base_label = int(d.y[indices[i]])
            adv_score = at.eval(adv_example)[0]
            adv_label = int(adv_score > 0.0)
            if adv_label != base_label:
                advs.append({
                    "index": indices[i],
                    "time": dur,
                    "adv_example": adv_example,
                    "base_example": example,
                    "base_label": base_label,
                    "adv_score": adv_score,
                    "eps": eps,
                    "linf": linf(example, adv_example)
                })
            else:
                base_score = at.eval(example)[0]
                fail_count += 1
                print("CUBE no adversarial example found", indices[i], base_label,
                      f"({base_score:.3f} -> {adv_score:.3f}, "\
                      f"nsuccess={len(advs)}, nfails={fail_count}/{len(indices)-len(advs)})")

            i += 1

            if len(advs) >= N: break

        if offset < len(advs): # new ones were generated
            for adv in advs[offset:]:
                adv["time"] /= (len(advs)-offset)
            dur = 0.0

    return advs

