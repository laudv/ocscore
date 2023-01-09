import json
import io
import numpy as np
import veritas
from subprocess import Popen, PIPE, STDOUT
from util import *

LT_ATTACK_EXEC = "../tree-ensemble-attack/lt_attack"

LT_ATTACK_DEFAULT_CONFIG = {
  "num_threads": 1,
  "enable_early_return": False,
  "inputs": "-",
  "model": "-",
  "num_classes": 2,
  "num_features": -1,
  "feature_start": 0,
  "num_point": -1,
  "num_attack_per_point": 20,
  "norm_type": 2,
  "search_mode": "lt-attack"
}

def lt_attack_config(at, **kwargs):
    used_features = sorted(at.get_splits().keys())
    num_features = used_features[-1] + 1

    config = LT_ATTACK_DEFAULT_CONFIG.copy()
    config["num_features"] = num_features
    for k, v in kwargs.items():
        config[k] = v
    return config

def _parse_adv_example_str(base, value):
    adv_ex = base.copy()
    splits = value.split(" ")
    pred_y = int(splits[0])
    adv_y = int(splits[1])
    for x in splits[2:]:
        feat_id, value = x.split(":")
        feat_id, value = int(feat_id), float(value)
        adv_ex[feat_id] = value
    return pred_y, adv_y, adv_ex

def _call_lt_attack(at, X, y, debug=False):
    p = Popen([LT_ATTACK_EXEC, "-"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)    
    conf = lt_attack_config(at,
            num_attack_per_point=20)

    s = io.StringIO()
    s.write(json.dumps(conf))
    s.write("\n<break>\n")
    s.write(at.to_json()) # write at
    s.write("\n<break>\n")

    for i in range(X.shape[0]):
        s.write(str(int(y[i]))) # write example
        for j in range(conf["num_features"]):
            s.write(f" {j}:{X[i,j]}")
        s.write("\n<break>\n")

    adv_examples = []
    adv_times = []

    outs, errs = p.communicate(s.getvalue().encode("ascii"))
    for line in outs.decode().splitlines():
        i = len(adv_examples)
        if line.startswith("<"):
            key, value = line[1:].split(">")
            if key == "FAIL":
                adv_examples.append("FAIL")
            elif key == "ADV_EX": # <true label> <adv label> [<feat_id>:<value>]+
                base_example = X[i, :]
                pred_y, adv_y, adv_example = _parse_adv_example_str(base_example, value)
                #print("LABELS from LT-attack:", pred_y, adv_y, "true label", int(y[i]))#, adv_example)
                verify_example_outcome(at, base_example, pred_y, f"LT normal example {i} label doesn't match")
                verify_example_outcome(at, adv_example, adv_y, f"LT adv example {i} label doesn't match")
                adv_examples.append(adv_example)
            elif key == "TIME":
                adv_times.append(float(value))
            elif key == "INCORRECT_PREDICTION":
                print("INCORRECT", value)
                adv_examples.append("INCORRECT")
                adv_times.append(0.0)
            else:
                raise RuntimeError(f"Don't know what `{key}` with value `{value}` is?")
        elif debug:
            print("DEBUG:", line)
                
    assert len(adv_examples) == X.shape[0], f"X.shape[0]={X.shape[0]}, len={len(adv_examples)}"
    assert len(adv_times) == X.shape[0]

    return adv_examples, adv_times

def get_adversarial_examples(d, indices, at, N, debug=False):
    if N > len(indices):
        print(f"WARNING: reducing N from {N} to {len(indices)}")
        N = len(indices)

    chunk_size = 100
    advs = []
    i = 0

    while len(advs) < N:
        r = indices[i:min(len(indices), i+chunk_size)]
        if len(r) == 0: break
        #print("CHUNK", r)
        Xsub = d.X.iloc[r, :].to_numpy()
        ysub = d.y[r].to_numpy()

        adv_examples, adv_times = _call_lt_attack(at, Xsub, ysub, debug)

        # remove fails
        for adv_example, time in zip(adv_examples, adv_times):
            example = d.X.iloc[indices[i], :].to_numpy()
            if not isinstance(adv_example, str):
                advs.append({
                    "index": indices[i],
                    "time": time,
                    "adv_example": adv_example,
                    "base_example": example,
                    "base_label": int(d.y[indices[i]]),
                    "adv_score": at.eval(adv_example)[0],
                    "linf": linf(example, adv_example)
                })
            else:
                print(f"LT-ATTACK {adv_example} for index {indices[i]}")

            i += 1

            if len(advs) >= N: break

    return advs

if __name__ == "__main__":
    at = veritas.AddTree()
    t = at.add_tree()
    t.split(0, 0, 1.0)
    t.set_leaf_value(t.left(t.root()), -1.2)
    t.set_leaf_value(t.right(t.root()), 2.2)
    print(t)

    X = np.array([[0.0], [2.0]])
    y = np.array([0, 1])

    advs = get_adversarial_examples(at, X, y, 10)
    print(advs)

    print(at.eval([[0.5], [2.5]]))
