import click
import numpy as np
import pandas as pd

import data_and_trees
from datasets import get_dataset, parse_dataset
from util import *
from adversarial_examples import *

from pprint import pprint

@click.command()
@click.argument("dataset", type=click.UNPROCESSED, callback=get_dataset)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "groot"]), default="xgb")
@click.option("-N", "N", default=100)
@click.option("--ratio", default=5)
@click.option("--fold", default=0)
@click.option("--nfolds", default=NFOLDS)
@click.option("--cache_dir", default="cache", show_default=True)
@click.option("--debug", default=None)
@click.option("--seed", default=SEED)
def per_dataset_perf(dataset, model_type, N, ratio, fold, nfolds, cache_dir, debug, seed):
    d, num_trees, tree_depth, lr = dataset
    d.seed = seed
    d.nfolds = nfolds
    d.load_dataset()
    d.minmax_normalize()

    model, meta, at = get_model(d, model_type, fold, lr, num_trees, tree_depth,
            groot_epsilon=INPUT_DELTA[d.name()])

    report = {"model_meta":  meta}
    report_name = get_report_name(d, seed, fold, N, ratio, model_type,
                                  num_trees, tree_depth, lr, cache_dir,
                                  special="")
    adv_name_template = get_adv_filename(d, seed, fold, N, model_type,
                                         num_trees, tree_depth, lr, cache_dir)

    refset0, refset1, refset_time = get_refset(d, at, fold)
    report["refset_time"] = refset_time

    if debug is not None:
        print("DEBUGGING INDEX", debug)
        indices = np.array([int(debug)])
        example = d.X.iloc[int(debug), :]
        print("at.eval", at.eval(example)[0], "with label", d.y[int(debug)])
        print(example)
        splits = at.get_splits()
        keys = sorted(splits.keys())
        for k in keys:
            vs = splits[k]
            x = example.iloc[k]
            m = np.argmin(np.array(vs)-x)
            print(" -", k, x, vs[m])

        print(at.base_score, len(at))


    else:
        indices = get_correct_test_example_indices(d, at, fold)

    lt_fname, kan_fname, ver_fname, cub_fname = get_adversarial_fnames(
            adv_name_template)
    report["lt_fname"] = lt_fname
    report["kan_fname"] = kan_fname
    report["ver_fname"] = ver_fname
    report["cub_fname"] = cub_fname
    lt_advs, kan_advs, ver_advs, cub_advs = get_adversarial_examples(
            adv_name_template, d, at, indices, N,
            cache=debug is None,
            cube_multiplier=10.0 if "Mnist" in d.name() or "Spambase" in d.name() else 5.0,
            cube_ntrials=1000)

    #for k1 in ["index", "adv_score", "linf", "time"]:
    #    print("==", k1, "==")
    #    for (k2, advs) in zip(["lt ", "kan", "ver", "cub"], [lt_advs, kan_advs, ver_advs, cub_advs]):
    #        print(k2, np.round([x[k1] for x in advs], 4))
    
    #analyze_advs_boxes(d, at, ver_advs[:5])
    #if "Mnist" in d.name():
    #    plot_mnist(at, ver_advs[:5])


    ### DETECTION ###

    # A second set of original test set examples
    rng = np.random.default_rng(29*fold + 3*nfolds + seed)
    min_fold_size = min(map(len, d.Ifolds))
    sample_indices = rng.choice(indices, min(N*ratio, min_fold_size))

    Xsample = d.X.iloc[sample_indices, :].to_numpy()
    ysample = d.y[sample_indices].to_numpy()
    report["Nadv"] = N
    report["Nsample"] = len(sample_indices)
    report["Ntotal"] = N + len(sample_indices)
    report["ratio"] = ratio
    report["sample_indices"] = sample_indices

    iforest, iforest_meta = d.get_iforest(fold)
    #lof = None
    lof, lof_meta = get_lof(d, fold, max_ntrain=20000)
    for adv_set, advs in zip(["lt", "kan", "ver", "cub"],
                               [lt_advs, kan_advs, ver_advs, cub_advs]):
        report[adv_set] = {}
        print("NUMBER OF ADV EXAMPLES", adv_set, len(advs), "vs", len(sample_indices), "NORMALS")

        Xadv = np.array([x["adv_example"] for x in advs])
        Xall = np.vstack((Xsample, Xadv))
        ypred = at.eval(Xall) > 0.0

        # what is the accuracy on the sample, with all advs considered incorrect?
        ycorrect = np.hstack((ypred[:len(ysample)]==ysample, np.zeros(N, dtype=bool)))

        # which examples should be detect as adversarial? => this is just zeros
        # for normals, ones for adversarials
        ydetect = np.hstack((np.zeros(len(ysample), dtype=bool), np.ones(N, dtype=bool)))

        report[adv_set][f"Nadv_{adv_set}"] = len(advs)
        report[adv_set]["sample_acc"] = np.mean(ypred[:len(ysample)] == ysample)
        report[adv_set]["ycorrect"] = ycorrect
        report[adv_set]["ydetect"] = ydetect

        # OC-score
        t = time.time()
        ps = ocscore.mapids(at, Xall)
        S0 = ocscore.ocscores(refset0, ps[~ypred])
        S1 = ocscore.ocscores(refset1, ps[ ypred])
        t = time.time() - t
        S = np.zeros(Xall.shape[0])
        S[~ypred] = S0
        S[ ypred] = S1
        stats = collect_stats(ydetect, ycorrect, S)
        stats["time"] = t
        report[adv_set]["ocscore"] = stats

        # AMBIGUITY
        t = time.time()
        pred_prob = at.predict_proba(Xall)
        ambig = 1.0 - np.abs((pred_prob * 2.0) - 1.0)
        t = time.time() - t
        stats = collect_stats(ydetect, ycorrect, ambig)
        stats["time"] = t
        report[adv_set]["ambig"] = stats

        # ISOLATION FOREST, w/o retraining
        report["iforest_time"] = iforest_meta["training_time"]
        t = time.time()
        S = iforest.score_samples(pd.DataFrame(Xall, columns=meta["columns"]))
        t = time.time() - t
        S = (S.max() - S)
        stats = collect_stats(ydetect, ycorrect, S)
        stats["time"] = t
        report[adv_set]["iforest"] = stats

        # LOCAL OUTLIER FACTOR
        if lof is not None:
            lof.novelty = True
            report["lof_time"] = lof_meta["training_time"]
            t = time.time()
            S = lof.score_samples(pd.DataFrame(Xall, columns=meta["columns"]))
            S = (S.max() - S)
            t = time.time() - t
            stats = collect_stats(ydetect, ycorrect, S)
            stats["time"] = t
            report[adv_set]["lof"] = stats

        # ML-LOO: https://ojs.aaai.org//index.php/AAAI/article/view/6140
        t = time.time()
        Sloo = np.zeros(ypred.shape)
        for i in range(Xall.shape[0]):
            x = Xall[i, :]
            xx = np.tile(x, (Xall.shape[1], 1))
            np.fill_diagonal(xx, 0.0)
            ypred_prob = at.predict_proba(x)[0]
            feat_attr = at.predict_proba(xx) - ypred_prob
            Sloo[i] = np.std(feat_attr)
        t = time.time() - t
        stats = collect_stats(ydetect, ycorrect, Sloo)
        stats["time"] = t
        report[adv_set]["mlloo"] = stats

        ## advclass: train a classifier to detect adversarial examples
        #advclass_at = get_advclass_model(d, seed, fold, nfolds, N, model_type,
        #        num_trees, tree_depth, lr, at, med_linf_rf, cache_dir)
        #t = time.time()
        #S = advclass_at.predict_proba(Xall)
        ##print(S[~ydetect])
        ##print(S[ydetect])
        #print(np.mean(S[~ydetect]))
        #print(np.mean(S[ydetect]))
        #t = time.time() - t
        #stats = collect_stats(ydetect, ycorrect, S)
        #stats["time"] = t
        #report[adv_set]["advclass"] = stats
        #print("advclass auc", stats["auc"])

    print("AUCs")
    print(["ambig", "ocscore", "iforest", "lof", "mlloo"])
    for adv_set in ["lt", "kan", "ver", "cub"]:
        print(f"{adv_set:4s}", end="")
        for detect_alg in ["ambig", "ocscore", "iforest", "lof", "mlloo"]:
            print(f"{report[adv_set][detect_alg]['auc']:.4f}", end=" ")
        print()

    dump(report_name, report)

#@click.command()
#@click.argument("dataset", type=click.UNPROCESSED, callback=get_dataset)
#@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "groot"]), default="xgb")
#@click.option("-N", "N", default=100)
#@click.option("--ratio", default=5)
#@click.option("--fold", default=0)
#@click.option("--nfolds", default=NFOLDS)
#@click.option("--cache_dir", default="cache", show_default=True)
#@click.option("--seed", default=SEED)
def vary_refset_size(dataset, model_type, N, ratio, fold, nfolds, cache_dir, seed):
    rng = np.random.default_rng(seed*79+97)
    #subsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    subsets = [0.1, 0.25, 0.5, 0.75, 1.0]
    adv_sets = ["lt", "kan", "ver", "cub"]

    d, num_trees, tree_depth, lr = dataset
    d.seed = seed
    d.nfolds = nfolds
    d.load_dataset()
    d.minmax_normalize()

    model, meta, at = get_model(d, model_type, fold, lr, num_trees, tree_depth,
            groot_epsilon=INPUT_DELTA[d.name()])

    report = {
        #"model_meta": meta,
        "model_type": model_type,
        "subsets": subsets,
        "aucs": [],
        "times": []
    }

    adv_name_template = get_adv_filename(d, seed, fold, N, model_type,
                                         num_trees, tree_depth, lr, cache_dir)

    full_refset0, full_refset1, refset_time = get_refset(d, at, fold)

    lt_fname, kan_fname, ver_fname, cub_fname = get_adversarial_fnames(
            adv_name_template)

#    report["lt_fname"] = lt_fname
#    report["kan_fname"] = kan_fname
#    report["ver_fname"] = ver_fname
#    report["cub_fname"] = cub_fname

    indices = get_correct_test_example_indices(d, at, fold)
    alladvs = get_adversarial_examples(
            adv_name_template, d, at, indices, N,
            cache=True,
            cube_multiplier=10.0 if "Mnist" in d.name() else 5.0)

    # A second set of original test set examples
    rng = np.random.default_rng(29*fold + 3*nfolds + seed)
    min_fold_size = min(map(len, d.Ifolds))
    sample_indices = rng.choice(indices, min(N*ratio, min_fold_size))

    Xsample = d.X.iloc[sample_indices, :].to_numpy()
    ysample = d.y[sample_indices].to_numpy()
    report["Nadv"] = N
    report["Nsample"] = len(sample_indices)
    report["Ntotal"] = N + len(sample_indices)
    report["ratio"] = ratio
    report["sample_indices"] = sample_indices

    Xadv = np.vstack([[x["adv_example"] for x in advs] for advs in alladvs])
    Xall = np.vstack((Xsample, Xadv))
    ypred = at.eval(Xall) > 0.0

    # what is the accuracy on the sample, with all advs considered incorrect?
    ycorrect = np.hstack((ypred[:len(ysample)]==ysample, np.zeros(Xadv.shape[0], dtype=bool)))

    # which examples should be detect as adversarial? => this is just zeros
    # for normals, ones for adversarials
    ydetect = np.hstack((np.zeros(len(ysample), dtype=bool), np.ones(Xadv.shape[0], dtype=bool)))

    for u, ss in enumerate(subsets):
        js = rng.choice(range(full_refset0.shape[0]),
                        int(full_refset0.shape[0]*ss))
        refset0 = full_refset0[js]
        js = rng.choice(range(full_refset1.shape[0]),
                        int(full_refset1.shape[0]*ss))
        refset1 = full_refset1[js]

        # OC-score
        t = time.time()
        ps = ocscore.mapids(at, Xall)
        S0 = ocscore.ocscores(refset0, ps[~ypred])
        S1 = ocscore.ocscores(refset1, ps[ ypred])
        t = time.time() - t
        S = np.zeros(Xall.shape[0])
        S[~ypred] = S0
        S[ ypred] = S1
        stats = collect_stats(ydetect, ycorrect, S)

        report["times"].append(t)
        report["aucs"].append(stats["auc"])

    return report

@click.command()
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "groot"]), default="xgb")
@click.option("-N", "N", default=100)
@click.option("--ratio", default=5)
@click.option("--nfolds", default=NFOLDS)
@click.option("--cache_dir", default="cache", show_default=True)
@click.option("--seed", default=SEED)
def vary_refset_size_combined(model_type, N, ratio, nfolds, cache_dir, seed):

    vary_refset_report = {}

    dnames = ["phoneme", "covtype", "mnist2v4", "ijcnn1", "webspam", "calhouse", "fmnist2v4"]
    for dname in dnames:
        dataset = parse_dataset(dname)
        vary_refset_report[dname] = []
        for fold in range(nfolds):
            report = vary_refset_size(dataset, model_type, N, ratio, fold, nfolds, cache_dir, seed)
            vary_refset_report[dname].append(report)

    dump(os.path.join(cache_dir, f"vary_refszet_size_{model_type}.joblib"), vary_refset_report)

if __name__ == "__main__":
    per_dataset_perf()


    # python experiments.py --model_type xgb --cache_dir cache_pinacs -N 500 --ratio 4
    #vary_refset_size_combined()



