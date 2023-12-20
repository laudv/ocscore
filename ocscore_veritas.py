import numpy as np
import ocscore
import veritas

# leaf2id[tree_id, node_id] = leaf ID
def get_leaf2id(at_or_tree, dtype):
    if isinstance(at_or_tree, veritas.AddTree):
        at = at_or_tree
        per_tree = [get_leaf2id(at[i], dtype) for i in range(len(at))]
        max_width = max(map(len, per_tree))
        leaf2id = np.zeros((len(per_tree), max_width), dtype=dtype) - 1 
        for i, l2id in enumerate(per_tree):
            leaf2id[i, 0:len(l2id)] = l2id
        return leaf2id
    else: # veritas.Tree
        tree = at_or_tree
        leaf2id = {leaf_id: i for i, leaf_id in enumerate(tree.get_leaf_ids())}
        if dtype == np.uint8 and len(leaf2id) > 2**8-1:
            raise RuntimeError("too many leafs for u8")
        if dtype == np.uint16 and len(leaf2id) > 2**16-1:
            raise RuntimeError("too many leafs for u16")
        leaf2id_arr = np.zeros(max(leaf2id.keys())+1, dtype=dtype) - 1
        for k, v in leaf2id.items():
            leaf2id_arr[k] = v
        return leaf2id_arr

def mapids(at, X, dtype=np.uint8):
    leafs = np.column_stack([at[i].eval_node(X) for i in range(len(at))])
    ids = ocscore.mapids(leafs, get_leaf2id(at, dtype), dtype)
    return ids
