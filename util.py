import os
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import ocscore
import ocscore_veritas
import veritas

from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor

SEED = 10
NFOLDS = 5

USED_DATASETS = ["calhouse", "electricity", "covtype", "higgs",
                 "ijcnn1", "mnist2v4", "fmnist2v4", "webspam"]

INPUT_DELTA = {
    "Phoneme":   0.05,
    "Spambase":  0.05,
    "CalhouseClf": 0.02,
    "Electricity": 0.02,
    "CovtypeNormalized":   0.08,
    "Higgs":     0.04,
    "Ijcnn1":    0.04,
    "MnistBinClass2v4":  0.4,# * 255,
    "FashionMnistBinClass2v4": 0.4,# * 255,
    "Webspam":   0.04,
}

MAX_TIME = {
    "Phoneme":   5,
    "Spambase":  5,
    "CalhouseClf": 10,
    "Electricity": 10,
    "CovtypeNormalized":   20,
    "Higgs":     30,
    "Ijcnn1":    10,
    "MnistBinClass2v4":  10,
    "FashionMnistBinClass2v4": 10,
    "Webspam":   10,
}

CUBE_MULTIPLIER = {
    "Phoneme":   5.0,
    "Spambase":  5.0,
    "CalhouseClf": 5.0,
    "Electricity": 5.0,
    "CovtypeNormalized":   5.0,
    "Higgs":     5.0,
    "Ijcnn1":    5.0,
    "MnistBinClass2v4":  20.0,
    "FashionMnistBinClass2v4": 10.0,
    "Webspam":   5.0,
}

CUBE_NTRIALS = {
    "Phoneme":   1000,
    "Spambase":  1000,
    "CalhouseClf": 1000,
    "Electricity": 1000,
    "CovtypeNormalized":   1000,
    "Higgs":     1000,
    "Ijcnn1":    1000,
    "MnistBinClass2v4": 2500,
    "FashionMnistBinClass2v4": 2000,
    "Webspam":   1000,
}

def fmt_val_std(v, e, best):
    sv = f"{v:1.2f}".lstrip("0")
    se = f"{e:1.2f}".lstrip("0")

    if sv == "1.00":
        sv = "1.0"

    sb = "\\bf" if abs(v-best)<0.01 else ""

    return f"{sb}{sv}{{\\tiny±{se}}}"

def dump(fname, data):
    joblib.dump(data, fname, compress=True)
    print(f"Results written to {fname}")

def load(fname):
    print(f"Reading {fname}")
    return joblib.load(fname)

def get_report_name(d, seed, fold, N, ratio, model_type, num_trees, tree_depth, lr, cache_dir, special=""):
    report_name = os.path.join(cache_dir, 
            f"{special}report_"
            f"{d.name()}-seed{seed}-fold{fold}_"
            f"N{N}:{ratio}_"
            f"{model_type}{num_trees}-{tree_depth}-{lr*100:.0f}.joblib")
    return report_name

def get_adv_filename(d, seed, fold, N, model_type, num_trees, tree_depth, lr, cache_dir):
    filename = os.path.join(cache_dir,
               f"<REPLACE>_{d.name()}_"\
               f"seed{seed}_"\
               f"fold{fold}-{d.nfolds}_N{N}_"\
               f"{model_type}{num_trees}-{tree_depth}-{lr*100:.0f}.joblib")
    return filename

def get_model(d, model_type, fold, lr, num_trees, tree_depth, groot_epsilon=None):
    d.nthreads = 1
    if model_type == "xgb":
        model, meta = d.get_xgb_model(fold, lr, num_trees, tree_depth)
    elif model_type == "rf":
        model, meta = d.get_rf_model(fold, num_trees, tree_depth=None)
    elif model_type == "groot":
        if groot_epsilon is None:
            raise RuntimeError("pass epsilon for groot")
        tree_depth = 8 # override
        model, meta = d.get_groot_model(fold, num_trees, tree_depth, groot_epsilon)
    else: raise RuntimeError(f"invalid model type {model_type}")
    at = d.get_addtree(model, meta)
    print(f"{model_type} model {meta['metric']}")
    return model, meta, at

def get_refset(d, at, fold):
    xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold)
    xtrain, ytrain = xtrain.to_numpy(), ytrain.to_numpy()
    #xtest, ytest = xtest.to_numpy(), ytest.to_numpy()

    # Only consider correctly classified examples in reference set
    tstart = time.time()
    ytrain_pred = at.eval(xtrain) > 0.0
    train_mask = ytrain_pred==ytrain
    xtrain_correct = xtrain[train_mask]
    ytrain_correct = ytrain[train_mask]
    ytrain_pred_correct = ytrain_pred[train_mask]

    xtrain_class0 = xtrain_correct[ytrain_correct == 0]
    xtrain_class1 = xtrain_correct[ytrain_correct == 1]

    refset_class0 = ocscore_veritas.mapids(at, xtrain_class0)
    refset_class1 = ocscore_veritas.mapids(at, xtrain_class1)

    refset_time = time.time() - tstart

    return refset_class0, refset_class1, refset_time

def get_correct_test_example_indices(d, at, fold):
    xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold)
    ytest_pred = at.eval(xtest) > 0.0
    test_mask = ytest==ytest_pred
    indices = np.copy(xtest.index[test_mask])

    return indices

def verify_example_outcome(at, example, expected_y, msg):
    y_score = at.eval(example)[0]
    y_at = int(y_score > 0.0)
    if y_score == 0.0: # tie breaker RFs
        print("TIE BREAKER!! y_score=0.0")
        y_at = expected_y
    assert y_at == expected_y, \
            f"{msg}: y_at {y_at} (score={y_score:.3f}), " \
            f"expected_y {expected_y} for example {example}"

def linf(ex1, ex2):
    return np.max(np.abs(ex1 - ex2))

def plot_mnist(at, advs):
    for adv in advs:
        base_ex = adv["base_example"].astype(np.float32)
        adv_ex = adv["adv_example"].astype(np.float32)
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.set_title(f"{at.eval(base_ex)[0]:.3f}")
        ax0.imshow(base_ex.reshape((28, 28)))
        ax1.set_title(f"{at.eval(adv_ex)[0]:.3f}")
        ax1.imshow(adv_ex.reshape((28, 28)))
        ax2.set_title(f"{adv['linf']:g}")
        im = ax2.imshow((adv_ex-base_ex).reshape((28, 28)))
        #fig.colorbar(im, ax=ax2)

    plt.show()

def analyze_advs_boxes(d, at, advs):
    print("BOX ANALYSIS")
    for adv in advs:
        print("|")
        print("| EXAMPLE")
        for m, t in enumerate(at):
            nbase = t.eval_node(adv["base_example"].astype(np.float32))
            nadv = t.eval_node(adv["adv_example"].astype(np.float32))

            if nbase != nadv:
                print(f"| different leaf for tree {m}")

                box_base = t.compute_box(nbase)
                box_adv = t.compute_box(nadv)

                for feat_id in range(d.X.shape[1]):
                    dom_base = box_base.get(feat_id, veritas.Domain())
                    dom_adv = box_adv.get(feat_id, veritas.Domain())
                    xbase = adv["base_example"][feat_id]
                    xadv = adv["adv_example"][feat_id]
                    if dom_base != dom_adv and xbase != xadv:
                        print(f"|   - {xbase:.12f} {xadv:.12f} {xbase-xadv:g}",
                              feat_id, dom_base, dom_adv)

def collect_stats(ydetect, ycorrect, S):
    perm = np.argsort(S)
    NN = np.sum(ydetect) # number of negatives = #adv. examples
    NP = len(ydetect)-NN # number of positives = #normal examples

    # coverage vs. accuracy of accepted examples
    cov = np.arange(len(ydetect)+1)
    cov_acc = np.ones(len(ydetect)+1)
    cov_acc[1:] = np.cumsum(ycorrect[perm]) / cov[1:]

    # TP: test set example accepted
    # FN: test set example rejected
    tp = np.zeros(len(ydetect)+1)
    tp[1:] = np.cumsum(1-ydetect[perm])
    fn = NP - tp # NP = TP+FN

    # FP: adv. example accepted
    # TN: adv. example rejected
    fp = np.zeros(len(ydetect)+1)
    fp[1:] = np.cumsum(ydetect[perm])
    tn = NN - fp # NN = FP+TN

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)

    #plt.plot(fpr0, tpr0)
    #plt.show()

    return { "S": S,
             "cov_acc": cov_acc,
             "auc": auc(fpr, tpr),
             "tpr": tpr,
             "fpr": fpr,
             "tnr": tnr,
             }

# From data_and_trees, but with limited training data, otherwise it takes too long
def get_lof(d, fold, max_ntrain):
    model_name = d.get_model_name(fold, "lof", 0, 0)
    model_path = os.path.join(d.model_dir, model_name)
    if os.path.isfile(model_path):
        print(f"loading LocalOutlierFactor from file: {model_name}")
        lof, meta = joblib.load(model_path)
    else:
        d.load_dataset()
        Xtrain, ytrain, Xtest, ytest = d.train_and_test_set(fold)
        if Xtrain.shape[0] > max_ntrain:
            print("get_lof: limiting training set size from",
                  Xtrain.shape[0], "to", max_ntrain)
            # already randomized in train_and_test_set
            Xtrain = Xtrain.iloc[0:max_ntrain,:]
            ytrain = ytrain[0:max_ntrain]

        t = time.time()
        lof = LocalOutlierFactor(n_neighbors=10)
        lof.fit(Xtrain)
        t = time.time() - t
        print(f"trained LocalOutlierFactor in {t:.2f}s");
        meta = {"training_time": t}
        joblib.dump((lof, meta), model_path)
    return lof, meta

def configure_matplotlib():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",

        "legend.frameon": False,
        "legend.fancybox": False,
        #"font.size": 6,
        #"axes.linewidth": 0.5,
        #"xtick.major.width": 0.5,
        #"ytick.major.width": 0.5,
        #"xtick.minor.width": 0.5,
        #"ytick.minor.width": 0.5,
        #"lines.linewidth": 0.6,

        "svg.fonttype": "none",

        "font.size": 7,
        "axes.linewidth":    0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth":   0.8,

        "hatch.linewidth": 0.5,

        #"text.latex.unicode" : False,
    })
    plt.rc("text.latex", preamble=r"\\usepackage{amsmath}")

# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 0.8, 0.6)
