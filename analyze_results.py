import os
import shutil
import subprocess
import click
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import pandas as pd
import numpy as np

from util import *
from datasets import THE_DATASETS, parse_dataset

from pprint import pprint
from sklearn.metrics import roc_auc_score, auc
from sklearn.neighbors import KernelDensity

DETECT_ALGS = ["ocscore", "ambig", "iforest", "mlloo"]
ADV_SETS = ["lt", "kan", "ver", "cub"]

CONF_BUCKETS = np.linspace(0.0, 1.0, 4)
CONF_BUCKETS[-1] = 1.0001 # make last conf_hi inclusive
CONF_BUCKETS = list(zip(CONF_BUCKETS, CONF_BUCKETS[1:]))

TABLE_DIR = os.environ["LATEX_TABLE_PATH"]
FIG_DIR = os.environ["LATEX_FIGURE_PATH"]

ALG_LS = {
    "ocscore": "-",
    "ambig": "--",
    "iforest": ":",
    "mlloo": "-.",
}
SET_HATCH = {
    "test": "",
    "lt": "|||",
    "kan": "///",
    "ver": "---",
    "cub": "\\\\\\",
}

SAVEFIGS=True


configure_matplotlib()

def collect_aucs_for_dataset(dataset, model_type, N, ratio, nfolds, cache_dir, seed):
    d, num_trees, tree_depth, lr = dataset
    d.seed = seed
    d.nfolds = nfolds
    #d.load_dataset()
    #d.minmax_normalize()

    aucs = {adv_set: {detect_alg: {"values": [], "mean": 0.0, "std": 0.0} for
                      detect_alg in DETECT_ALGS} for adv_set in ADV_SETS}

    print(d.name())
    for fold in range(nfolds):
        report_name = get_report_name(d, seed, fold, N, ratio, model_type,
                                      num_trees, tree_depth, lr, cache_dir,
                                      special="")
        if not os.path.exists(report_name):
            print("(!) skipping", report_name)
            continue

        report = load(report_name)
        for adv_set in ADV_SETS:
            for detect_alg in DETECT_ALGS:
                aucs[adv_set][detect_alg]["values"].append(report[adv_set][detect_alg]["auc"])
    for adv_set in ADV_SETS:
        for detect_alg in DETECT_ALGS:
            vs = aucs[adv_set][detect_alg]["values"]
            aucs[adv_set][detect_alg]["mean"] = np.mean(vs)
            aucs[adv_set][detect_alg]["std"] = np.std(vs)
    print()

    return aucs

def collect_aucs(model_type, N, ratio, nfolds, cache_dir, seed):
    aucs = {}
    for dname in THE_DATASETS:
        dataset = parse_dataset(dname)
        aucs[dname] = collect_aucs_for_dataset(dataset, model_type, N, ratio,
                                               nfolds, cache_dir, seed)
    return aucs


# per confidence bucket, compute AUC
def collect_aucs_per_conf_for_dataset(dataset, model_type, N, ratio, nfolds,
                                      cache_dir, seed):
    d, num_trees, tree_depth, lr = dataset
    d.seed = seed
    d.nfolds = nfolds
    d.load_dataset()
    d.minmax_normalize()

    aucs_per_conf = {detect_alg: [{"values": []} for i in
                                  range(len(CONF_BUCKETS))] for detect_alg in
                     DETECT_ALGS}

    for fold in range(nfolds):
        report_name = get_report_name(d, seed, fold, N, ratio, model_type,
                                      num_trees, tree_depth, lr, cache_dir,
                                      special="")
        if not os.path.exists(report_name):
            print("(!) skipping", report_name)
            continue

        model, meta, at = get_model(d, model_type, fold, lr, num_trees,
                                    tree_depth,
                                    groot_epsilon=INPUT_DELTA[d.name()])

        report = load(report_name)
        sample_indices = report["sample_indices"]
        Xsample = d.X.iloc[sample_indices, :].to_numpy()
        Xall = Xsample
        Sall = {}
        ydetect = np.zeros(len(sample_indices), dtype=bool)
        ysource = np.array(["test"] * len(sample_indices))

        for adv_set in ADV_SETS:
        #for adv_set in ["cub"]:
            advs_fname = report[f"{adv_set}_fname"]
            advs = load(advs_fname)

            Xadv = np.array([x["adv_example"] for x in advs])
            Xall = np.vstack((Xall, Xadv))
            ydetect = np.hstack((ydetect, np.ones(Xadv.shape[0], dtype=bool)))
            ysource = np.hstack((ysource, [adv_set] * Xadv.shape[0]))

            for detect_alg in DETECT_ALGS:
                S = report[adv_set][detect_alg]["S"]
                Nsample = report["Nsample"]

                if detect_alg not in Sall:
                    Sall[detect_alg] = S
                else:
                    Sall[detect_alg] = np.hstack((Sall[detect_alg], S[Nsample:]))

        assert len(ydetect) == Xall.shape[0]
        for detect_alg in DETECT_ALGS:
            assert len(Sall[detect_alg]) == Xall.shape[0]

        pred_prob = at.predict_proba(Xall)
        conf = np.abs((pred_prob * 2.0) - 1.0)

        for i, (conf_lo, conf_hi) in enumerate(CONF_BUCKETS):
            #conf_sample = (conf_lo <= conf) & (conf < conf_hi)
            #conf_sample = (conf_lo <= conf)
            conf_sample = (conf < conf_hi)

            if conf_sample.sum() == 0:
                print(f"SAMPLE EMPTY for conf [{conf_lo:.2f}, {conf_hi:.2f})")

            ydetect_sample = ydetect[conf_sample]
            ysource_sample = ysource[conf_sample]

            print(f"conf {conf_lo:.1f}",
                  "#normal", len(ydetect_sample),
                  "#adversarial", int(sum(ydetect_sample)))

            if np.all(ydetect_sample == ydetect_sample[0]):
                print(f"WARNING: all the same ydetect for conf [{conf_lo:.2f}, {conf_hi:.2f})",
                      ydetect_sample[0], d.name(), "fold", fold)

            for detect_alg in DETECT_ALGS:
                S = Sall[detect_alg][conf_sample]

                if np.all(ydetect_sample == ydetect_sample[0]):
                    auc = 1.0
                else:
                    auc = roc_auc_score(ydetect_sample, S)

                aucs_per_conf[detect_alg][i]["values"].append(auc)
                aucs_per_conf[detect_alg][i]["N"] = len(ydetect_sample)
                aucs_per_conf[detect_alg][i]["Nadv"] = int(sum(ydetect_sample))


    for i in range(len(CONF_BUCKETS)):
        for detect_alg in DETECT_ALGS:
            vs = aucs_per_conf[detect_alg][i]["values"]
            aucs_per_conf[detect_alg][i]["mean"] = np.mean(vs)
            aucs_per_conf[detect_alg][i]["std"] = np.std(vs)

    return aucs_per_conf

def collect_aucs_per_conf(model_type, N, ratio, nfolds, cache_dir, seed):
    aucs_per_conf = {}
    #for dname in THE_DATASETS:
    for dname in ["phoneme", "spambase", "mnist2v4", "calhouse"]:
        dataset = parse_dataset(dname)
        aucs_per_conf[dname] = collect_aucs_per_conf_for_dataset(dataset,
                                                                 model_type, N,
                                                                 ratio, nfolds,
                                                                 cache_dir,
                                                                 seed)
    return aucs_per_conf


def collect_accs_fixed_threshold_for_dataset(dataset, model_type, N, ratio,
                                             nfolds, cache_dir, seed):
    d, num_trees, tree_depth, lr = dataset
    d.seed = seed
    d.nfolds = nfolds
    d.load_dataset()
    d.minmax_normalize()

    accs = {detect_alg: {} for detect_alg in DETECT_ALGS}
    accs["nadvs"] = {}

    for fold in range(nfolds):
        report_name = get_report_name(d, seed, fold, N, ratio, model_type,
                                      num_trees, tree_depth, lr, cache_dir,
                                      special="")
        if not os.path.exists(report_name):
            print("(!) skipping", report_name)
            continue

        model, meta, at = get_model(d, model_type, fold, lr, num_trees,
                                    tree_depth,
                                    groot_epsilon=INPUT_DELTA[d.name()])

        report = load(report_name)
        sample_indices = report["sample_indices"]
        Nsample = len(sample_indices)
        Xsample = d.X.iloc[sample_indices, :].to_numpy()
        Xall = Xsample
        Xbase_all = None
        Sall = {}
        ydetect = np.zeros(len(sample_indices), dtype=bool)
        ysource = np.array(["test"] * len(sample_indices))

        # Step 1: collect all the S'sses for all the folds and for all
        # adversarial sets per detection method
        for adv_set in ADV_SETS:
            advs_fname = report[f"{adv_set}_fname"]
            advs = load(advs_fname)

            Xadv = np.array([x["adv_example"] for x in advs])
            Xbase = np.array([x["base_example"] for x in advs])
            Xall = np.vstack((Xall, Xadv))
            Xbase_all = np.vstack((Xbase_all, Xbase)) if Xbase_all is not None else Xbase
            ydetect = np.hstack((ydetect, np.ones(Xadv.shape[0], dtype=bool)))
            ysource = np.hstack((ysource, [adv_set] * Xadv.shape[0]))

            for detect_alg in DETECT_ALGS:
                S = report[adv_set][detect_alg]["S"]
                results[adv_set][detect_alg]["S"] = S
                if detect_alg not in Sall:
                    Sall[detect_alg] = S
                else:
                    Sall[detect_alg] = np.hstack((Sall[detect_alg], S[Nsample:]))

        assert len(ydetect) == Xall.shape[0]
        for detect_alg in DETECT_ALGS:
            assert len(Sall[detect_alg]) == Xall.shape[0]

        # Compute the confidence of the ensemble for each example
        pred_prob = at.predict_proba(Xall)
        conf = np.abs(pred_prob - 0.5) + 0.5
        conf_perm = np.argsort(conf)

        # Compute linf deltas between original examples and the adversarial examples
        delta = np.sum(np.abs(Xall[Nsample:, :]-Xbase_all), axis=1)

        # Step 2: Fix the threshold
        # fix threshold, then check accuracy for each point
        Nall = Xall.shape[0]
        sample_ratio = Nsample/Nall
        for detect_alg in DETECT_ALGS:
            S = Sall[detect_alg]
            threshold = np.quantile(S, sample_ratio)
            ydetect_pred = S > threshold
            ydetect_acc = ydetect_pred == ydetect

            window_size = 100
            #windows = np.arange(window_size)[np.newaxis, :]\
            #        + np.arange(len(S)-window_size+1)[:, np.newaxis]

            sliding_acc = np.lib.stride_tricks.sliding_window_view(
                    ydetect_acc[conf_perm], window_size).mean(axis=1)
            sliding_conf = np.lib.stride_tricks.sliding_window_view(
                    conf[conf_perm], window_size).mean(axis=1)

            #skipper = np.arange(1, len(sliding_acc), 4)
            #sliding_acc = sliding_acc[skipper]
            #sliding_conf = sliding_conf[skipper]

            #sliding_acc = np.convolve(sliding_acc, np.ones(5)/5, mode="same")

            # sliding window on confidence
            if "sliding_acc" not in accs[detect_alg]:
                accs[detect_alg]["sliding_acc"] = np.zeros((nfolds, len(sliding_acc)))
                accs[detect_alg]["sliding_conf"] = np.zeros((nfolds, len(sliding_acc)))
            accs[detect_alg]["sliding_acc"][fold, :] = sliding_acc
            accs[detect_alg]["sliding_conf"][fold, :] = sliding_conf

        # how many adversarial examples in the window?
        #sliding_nadv = np.lib.stride_tricks.sliding_window_view(
        #        ydetect[conf_perm], window_size).mean(axis=1)
        #sliding_conf = np.lib.stride_tricks.sliding_window_view(
        #        conf[conf_perm], window_size).mean(axis=1)
        #if "sliding_acc" not in accs["nadvs"]:
        #    accs["nadvs"]["sliding_acc"] = np.zeros((nfolds, len(sliding_acc)))
        #    accs["nadvs"]["sliding_conf"] = np.zeros((nfolds, len(sliding_acc)))
        #accs["nadvs"]["sliding_acc"][fold, :] = sliding_nadv
        #accs["nadvs"]["sliding_conf"][fold, :] = sliding_conf

        nadv = np.cumsum(ydetect[conf_perm]) / (Xall.shape[0]-Nsample) # how many adversarials have we seen?
        nadv = np.cumsum(np.ones(len(ydetect))) / Xall.shape[0] # how many examples have we seen?
        if "sliding_acc" not in accs["nadvs"]:
            accs["nadvs"]["sliding_acc"] = np.zeros((nfolds, len(nadv)))
            accs["nadvs"]["sliding_conf"] = np.zeros((nfolds, len(nadv)))
        accs["nadvs"]["sliding_acc"][fold, :] = nadv
        accs["nadvs"]["sliding_conf"][fold, :] = conf[conf_perm]

    for detect_alg in accs.keys():
        skipper = np.arange(1, accs[detect_alg]["sliding_acc"].shape[1], 5)
        accs[detect_alg]["acc_mean"] = np.mean(accs[detect_alg]["sliding_acc"], axis=0)[skipper]
        accs[detect_alg]["acc_std"] = np.std(accs[detect_alg]["sliding_acc"], axis=0)[skipper]
        accs[detect_alg]["conf"] = np.mean(accs[detect_alg]["sliding_conf"], axis=0)[skipper]

    return accs


def collect_accs_fixed_threshold(model_type, N, ratio, nfolds, cache_dir, seed):
    accs = {}
    #for dname in THE_DATASETS:
    #for dname in ["covtype", "spambase", "mnist2v4", "calhouse", "ijcnn1"]:
    for dname in ["covtype", "mnist2v4"]:
        dataset = parse_dataset(dname)
        accs[dname] = collect_accs_fixed_threshold_for_dataset(dataset,
                                                               model_type, N,
                                                               ratio, nfolds,
                                                               cache_dir, seed)
    return accs

def collect_results_dataset(dataset, model_type, N, ratio, nfolds, cache_dir, seed):
    d, num_trees, tree_depth, lr = dataset
    d.seed = seed
    d.nfolds = nfolds
    d.load_dataset()
    d.minmax_normalize()

    per_set_all = []
    per_alg_all = []
    per_set_alg_all = []
    conf_sample = []

    for fold in range(nfolds):
        per_alg = {detect_alg: {} for detect_alg in DETECT_ALGS}
        per_set = {adv_set: {} for adv_set in ADV_SETS}
        per_set_alg = {adv_set: {detect_alg: {}
                           for detect_alg in DETECT_ALGS}
                       for adv_set in ADV_SETS}

        per_alg_all.append(per_alg)
        per_set_all.append(per_set)
        per_set_alg_all.append(per_set_alg)

        report_name = get_report_name(d, seed, fold, N, ratio, model_type,
                                      num_trees, tree_depth, lr, cache_dir,
                                      special="")
        if not os.path.exists(report_name):
            print("(!) skipping", report_name)
            continue

        model, meta, at = get_model(d, model_type, fold, lr, num_trees,
                                    tree_depth,
                                    groot_epsilon=INPUT_DELTA[d.name()])

        report = load(report_name)
        sample_indices = report["sample_indices"]
        Nsample = len(sample_indices)
        Xsample = d.X.iloc[sample_indices, :].to_numpy()
        pred_prob_sample = at.predict_proba(Xsample)
        conf_sample.append(np.abs(pred_prob_sample - 0.5) + 0.5)
        Xadv, Xbase = {}, {}

        per_alg["ocscore"]["setup_time"] = report["refset_time"]
        per_alg["iforest"]["setup_time"] = report["iforest_time"]
        per_alg["ambig"]["setup_time"] = 0.0
        per_alg["mlloo"]["setup_time"] = 0.0
        if "lof_time" in report:
            per_alg["lof"]["setup_time"] = report["lof_time"]

        # per adv set
        for adv_set in ADV_SETS:
            advs_fname = report[f"{adv_set}_fname"]
            advs = load(advs_fname)

            Xadv[adv_set] = np.array([x["adv_example"] for x in advs]).astype(np.float32)
            Xbase[adv_set] = np.array([x["base_example"] for x in advs]).astype(np.float32)

            # DELTA
            absdiff = np.abs(Xadv[adv_set]-Xbase[adv_set]).max(axis=1)
            per_set[adv_set]["delta"] = absdiff

            # CONFIDENCE
            pred_prob = at.predict_proba(Xadv[adv_set])
            conf = np.abs(pred_prob - 0.5) + 0.5

            per_set[adv_set]["pred_prob"] = pred_prob
            per_set[adv_set]["conf"] = conf

        # collect aucs and scores S per adv_set and per algorithm
        for adv_set in ADV_SETS:
            for detect_alg in DETECT_ALGS:
                per_set_alg[adv_set][detect_alg]["auc"] = report[adv_set][detect_alg]["auc"]
                per_set_alg[adv_set][detect_alg]["time"] = report[adv_set][detect_alg]["time"]
                Sfull = report[adv_set][detect_alg]["S"]
                Sadv = Sfull[Nsample:]
                per_set_alg[adv_set][detect_alg]["S"] = Sadv
                sample_ratio_per_set = (Nsample+1)/len(Sfull)
                per_set_alg[adv_set][detect_alg]["threshold_per_set"] =\
                        np.quantile(Sfull, sample_ratio_per_set)

        # scores per algorithm on the test set sample
        for detect_alg in DETECT_ALGS:
            S = report["ver"][detect_alg]["S"][:Nsample]
            per_alg[detect_alg]["Ssample"] = S

        # determine fixed threshold on all adversarial examples for each detection algorithm
        Xall = np.vstack([Xsample] + list(Xadv.values()))
        is_adv = np.hstack([np.zeros(Nsample), np.ones(Xall.shape[0]-Nsample)])
        sample_ratio = (Nsample+1)/Xall.shape[0]
        for detect_alg in DETECT_ALGS:
            Ssample = per_alg[detect_alg]["Ssample"]
            Sall = np.hstack([Ssample] + [per_set_alg[s][detect_alg]["S"] for s in ADV_SETS])
            thrs = np.quantile(Sall, sample_ratio)
            per_alg[detect_alg]["threshold"] = thrs

            # collect aggregated stats over all 4 sets
            per_alg[detect_alg]["acc_aggr"] = np.mean(is_adv == (Sall >= thrs))
            per_alg[detect_alg]["auc_aggr"] = roc_auc_score(is_adv, Sall)

            per_alg[detect_alg]["is_adv_pred_sample"] = (Ssample >= thrs)

        # collect accs per adv_set and per algorithm using threshold
        for adv_set in ADV_SETS:
            for detect_alg in DETECT_ALGS:
                S = per_set_alg[adv_set][detect_alg]["S"]
                Ssample = per_alg[detect_alg]["Ssample"]
                thrs = per_alg[detect_alg]["threshold"]
                is_adv_pred = (S >= thrs)
                per_set_alg[adv_set][detect_alg]["is_adv_pred"] = is_adv_pred
                per_set_alg[adv_set][detect_alg]["acc"] = np.mean(is_adv_pred)
                thrs_per_set = per_set_alg[adv_set][detect_alg]["threshold_per_set"]
                per_set_alg[adv_set][detect_alg]["acc_per_set"] = np.mean(S >= thrs_per_set)
                per_set_alg[adv_set][detect_alg]["acc_sample_per_set"] = np.mean(Ssample < thrs_per_set)


        # performance on test set sample
        for detect_alg in DETECT_ALGS:
            S = per_alg[detect_alg]["Ssample"]
            thrs = per_alg[detect_alg]["threshold"]
            per_alg[detect_alg]["acc_sample"] = np.mean(S < thrs)

    ## END FOLDS FOR-LOOP

    # collect everything over the folds
    per_alg = {detect_alg: {} for detect_alg in DETECT_ALGS}
    per_set = {adv_set: {} for adv_set in ADV_SETS}
    per_set_alg = {adv_set: {detect_alg: {}
                       for detect_alg in DETECT_ALGS}
                   for adv_set in ADV_SETS}
    for adv_set in ADV_SETS:
        for detect_alg in DETECT_ALGS:
            for k in ["auc", "acc", "threshold_per_set", "acc_per_set",
                      "acc_sample_per_set", "time"]:
                per_set_alg[adv_set][detect_alg][f"{k}_mean"] =\
                        np.mean([x[adv_set][detect_alg][k] for x in per_set_alg_all])
                per_set_alg[adv_set][detect_alg][f"{k}_std"] =\
                        np.std([x[adv_set][detect_alg][k] for x in per_set_alg_all])
    for adv_set in ADV_SETS:
        per_set[adv_set]["delta_mean"] = np.mean([x[adv_set]["delta"].mean()
                                                  for x in per_set_all])
        per_set[adv_set]["delta_std"] = np.std([x[adv_set]["delta"].mean()
                                                for x in per_set_all])

    for detect_alg in DETECT_ALGS:
        for k in ["acc_sample", "threshold", "acc_aggr", "auc_aggr"]:
            per_alg[detect_alg][f"{k}_mean"] = np.mean([x[detect_alg][k].mean()
                                                      for x in per_alg_all])
            per_alg[detect_alg][f"{k}_std"] = np.std([x[detect_alg][k].mean()
                                                    for x in per_alg_all])

    # combine all is_adv_pred's over all folds, together with conf & delta
    conf_sample = np.hstack(conf_sample)
    delta_sample = np.zeros(conf_sample.shape)
    per_confdelta = {
        "conf": conf_sample,
        "delta": delta_sample,
        "set": ["test"] * len(conf_sample),
        "is_adv_pred": {
            detect_alg: np.hstack([pa[detect_alg]["is_adv_pred_sample"] for pa in per_alg_all])
            for detect_alg in DETECT_ALGS
        }
    }
    for fold in range(nfolds):
        ps = per_set_all[fold]
        psa = per_set_alg_all[fold]

        # adversarial SETS for fold
        for adv_set in ADV_SETS:
            set_arr = [adv_set] * len(ps[adv_set]["conf"])
            per_confdelta["conf"] = np.hstack((per_confdelta["conf"], ps[adv_set]["conf"]))
            per_confdelta["delta"] = np.hstack((per_confdelta["delta"], ps[adv_set]["delta"]))
            per_confdelta["set"] = np.hstack((per_confdelta["set"], set_arr))

            pred = per_confdelta["is_adv_pred"]

            for detect_alg in DETECT_ALGS:
                pred[detect_alg] = np.hstack((
                    pred[detect_alg],
                    psa[adv_set][detect_alg]["is_adv_pred"]))

    return per_set, per_alg, per_set_alg, per_confdelta

def collect_results(dnames, model_type, N, ratio, nfolds, cache_dir, seed):
    per_set, per_alg, per_set_alg, per_confdelta = {}, {}, {}, {}
    for dname in dnames:
    #for dname in ["phoneme", "covtype", "mnist2v4", "ijcnn1", "webspam", "calhouse"]:
    #for dname in ["mnist2v4", "covtype", "ijcnn1", "webspam"]:
    #for dname in ["phoneme", "mnist2v4"]:
        dataset = parse_dataset(dname)
        ps, pa, psa, pcd = collect_results_dataset(dataset, model_type, N,
                                                   ratio, nfolds, cache_dir,
                                                   seed)
        per_set[dname] = ps
        per_alg[dname] = pa
        per_set_alg[dname] = psa
        per_confdelta[dname] = pcd
    
    return per_set, per_alg, per_set_alg, per_confdelta


def display_results(per_set, per_alg, per_set_alg):
    dnames = list(per_set.keys())
    df_auc_per_alg = pd.DataFrame("-", index=THE_DATASETS, columns=DETECT_ALGS)
    df_acc_per_alg = pd.DataFrame("-", index=THE_DATASETS, columns=DETECT_ALGS)
    df_auc_per_set = pd.DataFrame("-", index=DETECT_ALGS, columns=ADV_SETS)

    set_kinds = ["low confidence", "high confidence"]
    df_auc_per_kind = pd.DataFrame("-", index=DETECT_ALGS, columns=set_kinds)

    index = pd.MultiIndex.from_product([DETECT_ALGS, ADV_SETS], names=["Algorithm", "Set"])
    df_auc = pd.DataFrame("-", index=index, columns=THE_DATASETS)
    df_time = pd.DataFrame("-", index=THE_DATASETS, columns=DETECT_ALGS)

    for dname in dnames:
        ps = per_set[dname]
        pa = per_alg[dname]
        psa = per_set_alg[dname]

        aucmax = max(pa[da]["auc_aggr_mean"] for da in DETECT_ALGS)
        aucmax_ps = {s:max(psa[s][da]["auc_mean"] for da in DETECT_ALGS) for s in ADV_SETS}

        for detect_alg in pa.keys():
            v = pa[detect_alg]["auc_aggr_mean"]
            e = pa[detect_alg]["auc_aggr_std"]
            best = "\\bf" if np.abs(v-aucmax)<0.01 else ""
            df_auc_per_alg.loc[dname, detect_alg] = f"{best}{v:1.2f}{{\\tiny±{e:1.2f}}}"
            df_acc_per_alg.loc[dname, detect_alg] = pa[detect_alg]["acc_aggr_mean"]

            v = np.sum([psa[a][detect_alg]["time_mean"] for a in psa.keys()]) / (5*2500) * 1000
            e = np.sum([psa[a][detect_alg]["time_std"] for a in psa.keys()]) / (5*2500) * 1000
            #df_time.loc[dname, detect_alg] = f"{v:1.2f}{{\\tiny±{e:1.2f}}}"
            df_time.loc[dname, detect_alg] = f"{v:1.3f}"

            for adv_set in ADV_SETS:
                v = psa[adv_set][detect_alg]["auc_mean"]
                e = psa[adv_set][detect_alg]["auc_std"]
                best = "\\bf" if np.abs(v-aucmax_ps[adv_set])<max(0.001, e) else ""
                df_auc.loc[(detect_alg, adv_set), dname] = f"{best}{v:1.2f}{{\\tiny±{e:1.2f}}}"

        for adv_set in pa.keys():
            pass

    # Per set performance of each alg, averaged over all datasets
    for adv_set in ADV_SETS:
        aucs = {}

        for detect_alg in DETECT_ALGS:
            vs = [per_set_alg[d][adv_set][detect_alg]["auc_mean"] for d in dnames]
            aucs[detect_alg] = vs

        aucmax = max(np.mean(v) for v in aucs.values())

        for detect_alg, vs in aucs.items():
            v = np.mean(aucs[detect_alg])
            e = np.std(aucs[detect_alg])
            best = "\\bf" if np.abs(v-aucmax)<0.01 else ""
            df_auc_per_set.loc[detect_alg, adv_set] = f"{best}{v:1.2f}{{\\tiny±{e:1.2f}}}"

    # Per set 'kind' performance of each alg, averaged over all datasets
    for kind, adv_sets in zip(set_kinds, [["lt", "kan"], ["ver", "cub"]]):
        aucs = {}

        for detect_alg in DETECT_ALGS:
            vs = []
            for adv_set in adv_sets:
                vs += [per_set_alg[d][adv_set][detect_alg]["auc_mean"]
                       for d in dnames]
            aucs[detect_alg] = vs

        aucmax = max(np.mean(v) for v in aucs.values())

        for detect_alg, vs in aucs.items():
            v = np.mean(aucs[detect_alg])
            e = np.std(aucs[detect_alg])
            best = "\\bf" if np.abs(v-aucmax)<0.01 else ""
            df_auc_per_kind.loc[detect_alg, kind] = f"{best}{v:1.2f}{{\\tiny±{e:1.2f}}}"

    print("\nauc_aggr AUC aggregated over all sets")
    print(df_auc_per_alg)
    with open(os.path.join(TABLE_DIR, "auc_aggr_table.tex"), "w") as f:
        #df_auc_per_alg.columns = [f"{{{x}}}" for x in df_auc_per_alg.columns]
        df_auc_per_alg.to_latex(buf=f, longtable=False, escape=False,
                column_format="l"+"l"*df_auc_per_alg.shape[1])

    print("\nauc_aggr AUC aggregated over all datasets")
    print(df_auc_per_set)
    print(df_auc_per_kind)
    with open(os.path.join(TABLE_DIR, "auc_aggr_per_set_table.tex"), "w") as f:
        #df_auc_per_alg.columns = [f"{{{x}}}" for x in df_auc_per_alg.columns]
        df_auc_per_kind.to_latex(buf=f, longtable=False, escape=False,
                column_format="l"+"c"*df_auc_per_kind.shape[1])

    print("\nDF AUC")
    print(df_auc)
    with open(os.path.join(TABLE_DIR, "auc_table.tex"), "w") as f:
        #df_auc_per_alg.columns = [f"{{{x}}}" for x in df_auc_per_alg.columns]
        df_auc.to_latex(buf=f, longtable=False, escape=False,
                column_format="ll"+"l"*df_auc.shape[1])

    print("\nDF TIME")
    print(df_time)
    with open(os.path.join(TABLE_DIR, "time_table.tex"), "w") as f:
        #df_auc_per_alg.columns = [f"{{{x}}}" for x in df_auc_per_alg.columns]
        df_time.to_latex(buf=f, longtable=False, escape=False,
                column_format="l"+"l"*df_time.shape[1])

def plot_confdelta(per_confdelta, per_alg):
    dnames = per_confdelta.keys()
    fig, axs = plt.subplots(2, len(dnames),
                            figsize=(6.2, 2.0),
                            sharex=True,
                            gridspec_kw={'height_ratios':[2, 1]})
    fig.subplots_adjust(left=0.07, bottom=0.17, right=0.98, hspace=0.2,
                        wspace=0.3, top=0.78)

    for ax, axl, dname in zip(axs[0, :], axs[1, :], dnames):
        pcd = per_confdelta[dname]
        pa = per_alg[dname]
        conf = pcd["conf"]
        #conf = pcd["delta"]

        #confq0 = np.linspace(min(conf), max(conf), 20)
        #confq1 = np.quantile(conf, np.linspace(0, 1, 101))
        #confq = np.sort(np.hstack((confq0, confq1)))
        confq = np.quantile(conf, np.linspace(0, 1, 201))
        confq[-1] += 0.001 # make inclusive
        intervals = list(zip(confq, confq[20:]))
        xs = []
        xs2 = [] # average conf in window instead of mid
        ns = []
        ns_adv = []
        ws = [] # number of ex in window
        is_adv = pcd["set"] != "test"
        n_adv = sum(is_adv)
        for vlo, vhi in intervals:
            vmid = vlo + (vhi-vlo)/2.0
            xs.append(vmid)
            mask = (conf < vhi)
            ns.append(mask.mean())
            ns_adv.append((mask & is_adv).sum() / n_adv)
            mask = (vlo <= conf) & (conf < vhi)
            ws.append(mask.mean())
            xs2.append(conf[mask].mean())
        ns[0] = 0
        #axl.plot(ns, xs, label="Confidence", color="gray", ls="--")
        axl.plot(ns, xs2, label="Confidence", color="gray", ls="-")
        #axlt = axl.twinx()
        #axlt.plot(ns, ns_adv, label="Fraction Adv", color="gray", ls=":")
        #axlt.set_ylim([0.0, 1.05])
        #axl.plot(xs, ns, label="#ex seen", color="gray", ls="-")
        #axl.plot(ns, ns_adv, label="#advs", color="lightgray", ls=":")
        #ax.plot(ns, ws, label="#window", color="lightgray", ls="-")
        #print("window sizes", dname, np.round(ws, 3))

        lines = []
        for detect_alg in DETECT_ALGS:
            threshold = 1.0 - 0.5*pa[detect_alg]["threshold_mean"]
            is_adv_pred = pcd["is_adv_pred"][detect_alg]
            is_correct = is_adv == is_adv_pred
            acc_per_conf = []
            for vlo, vhi in intervals:
                mask = (vlo <= conf) & (conf < vhi)
                acc_per_conf.append(is_correct[mask].mean())
            l, = ax.plot(ns, acc_per_conf, label=detect_alg, ls=ALG_LS[detect_alg])
            lines.append(l)

            #if detect_alg == "ambig":
            #    thrs_idx = np.argmin(np.maximum(0.0, threshold-xs2))
            #    print("thrs_idx", thrs_idx, xs2[thrs_idx], threshold)
            #    ax.plot([ns[thrs_idx]], [0.0], "^", c=l.get_color())


        #axl.set_xlabel("Confidence")
        axl.set_xlabel("fraction of examples")
        #xticks = np.linspace(0, 1, 5)
        #xticklabels = np.quantile(confq, xticks).round(2)
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(xticklabels)
        ax.set_title(dname)
        ax.set_xlim([0.0, 1.0])
        axl.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        axl.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axl.set_xticklabels([".0", ".2", ".4", ".6", ".8", "1"])
        ax.set_xticks(axl.get_xticks())
        #ax.set_xticklabels([""] * len(axl.get_xticks()))
        #ax.grid(visible=True, axis="x", which="major", color="lightgray")
        #axl.grid(visible=True, axis="x", which="major", color="lightgray")
        axl.set_ylim([0.5, 1.05])

        #fig, ax = plt.subplots()
        #ax.set_title(dname)
        #for adv_set in ADV_SETS:
        #    mask = pcd["set"] == adv_set
        #    mask0 = mask & (is_adv_pred == False)
        #    mask1 = mask & (is_adv_pred == True)
        #    x = pcd["conf"][mask0]
        #    y = pcd["delta"][mask0]
        #    s, = ax.plot(x, y, "x", label=adv_set)
        #    ax.set_xscale("log")
        #    ax.set_yscale("log")
        #    x = pcd["conf"][mask1]
        #    y = pcd["delta"][mask1]
        #    ax.plot(x, y, "d", label=adv_set, color=s.get_color())
        #ax.legend()

    #axs[0, 0].legend()
    fig.legend(handles=lines, ncol=len(lines), fancybox=False, loc="upper center")
    axs[0, 0].set_ylabel("Accuracy")
    #axs[1, 0].legend()
    axs[1, 0].set_ylabel("Conf.")

    if SAVEFIGS:
        figname = "acc_per_conf"
        fig.savefig(os.path.join("figures", f"{figname}.svg"))
        subprocess.run(["/home/laurens/repos/dotfiles/scripts/svg2latex", f"{figname}.svg"], cwd="figures")
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf_tex"),
                        os.path.join(FIG_DIR, f"{figname}.pdf_tex"))
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf"),
                        os.path.join(FIG_DIR, f"{figname}.pdf"))
    plt.show()

def plot_confdist(per_confdelta, model_type): # all in one plot --> busy hard to read
    dnames = per_confdelta.keys()
    #fig, axs = plt.subplots(1, len(dnames), figsize=(20, 5))
    #fig.subplots_adjust(left=0.02, right=0.98)
    cmap = plt.get_cmap("tab10")

    all_conf = np.array([])
    all_sets = np.array([])

    #for ax, dname in zip(axs.ravel(), dnames):
    for k, dname in enumerate(dnames):
        pcd = per_confdelta[dname]
        conf = pcd["conf"]
        sets = pcd["set"]

        all_conf = np.hstack((all_conf, conf))
        all_sets = np.hstack((all_sets, sets))

        bins = np.linspace(0.5, 1.0, 21)
        x = np.linspace(0.5, 1.0, 101)
        bottom = np.zeros(x.shape)

        #ax=axs.ravel()[k]
        #for u, s in enumerate(ADV_SETS + ["test"]):
        #    conf_s = conf[sets == s]
        #    hist, bin_edges = np.histogram(conf_s, bins=bins)
        #    hist = hist / len(conf)
        #    kwargs = {"facecolor": cmap(u)}
        #    if s == "test":
        #        kwargs["facecolor"] = "gray"
        #    kde = KernelDensity(kernel='gaussian', bandwidth=0.03).fit(conf_s.reshape(-1, 1))
        #    y = np.exp(kde.score_samples(x.reshape(-1, 1))) / (len(ADV_SETS)+1)
        #    #ax.bar(bins[:-1], hist, width=0.05, align="edge", bottom=bottom,
        #    #       label=s, hatch=SET_HATCH[s],
        #    #       linewidth=0.5, edgecolor="black", **kwargs)
        #    #ax.plot(bins[:-1], hist+bottom, label=s)
        #    ax.fill_between(x, bottom, bottom+y, label=s, edgecolor="black",
        #                    linewidth=0.5, hatch=SET_HATCH[s], **kwargs)
        #    #print(f"{s:3}", hist)
        #    bottom += y

        #ax.set_title(dname)
        #ax.set_xlabel("Confidence")
        #ax.set_ylabel("Density")
        #ax.legend()

    fig2, ax = plt.subplots(figsize=(1.8, 1.8))
    fig2.subplots_adjust(left=0.15, bottom=0.19, right=0.98, top=0.96)

    conf = all_conf
    sets = all_sets
    bottom = np.zeros(x.shape)
    for u, s in enumerate(ADV_SETS + ["test"]):
        conf_s = conf[sets == s]
        hist, bin_edges = np.histogram(conf_s, bins=bins)
        hist = hist / len(conf)
        kwargs = {"facecolor": cmap(u)}
        if s == "test":
            kwargs["facecolor"] = "gray"
        kde = KernelDensity(kernel='epanechnikov', bandwidth=0.02).fit(conf_s.reshape(-1, 1))
        #y = kde.score_samples(x.reshape(-1, 1))
        y = np.exp(kde.score_samples(x.reshape(-1, 1)))
        #ax.bar(bins[:-1], hist, width=0.05, align="edge", bottom=bottom,
        #       label=s, hatch=SET_HATCH[s],
        #       linewidth=0.5, edgecolor="black", **kwargs)
        #ax.plot(bins[:-1], hist+bottom, label=s)
        ax.fill_between(x, bottom, bottom+y, label=s, edgecolor="black",
                        linewidth=0.5, hatch=SET_HATCH[s], **kwargs)
        #print(f"{s:3}", hist)
        bottom += y

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.0, ax.get_ylim()[1]])
    ax.set_yticklabels([""] * len(ax.get_yticks()))
    ax.legend(loc="upper center")

    if SAVEFIGS:
        figname = f"conf_density_{model_type}"
        fig2.savefig(os.path.join("figures", f"{figname}.svg"))
        subprocess.run(["/home/laurens/repos/dotfiles/scripts/svg2latex", f"{figname}.svg"], cwd="figures")
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf_tex"),
                        os.path.join(FIG_DIR, f"{figname}.pdf_tex"))
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf"),
                        os.path.join(FIG_DIR, f"{figname}.pdf"))

    plt.show()

def plot_confdist2(per_confdelta, model_type): # 5 different subplots
    dnames = per_confdelta.keys()
    cmap = plt.get_cmap("tab10")

    all_conf = np.array([])
    all_sets = np.array([])

    for k, dname in enumerate(dnames):
        pcd = per_confdelta[dname]
        conf = pcd["conf"]
        sets = pcd["set"]

        all_conf = np.hstack((all_conf, conf))
        all_sets = np.hstack((all_sets, sets))

        bins = np.linspace(0.5, 1.0, 21)
        x = np.linspace(0.5, 1.0, 101)
        bottom = np.zeros(x.shape)

    fig, axs = plt.subplots(1, 5, figsize=(6.2, 0.9), sharey=True)
    fig.subplots_adjust(left=0.055, bottom=0.36, right=0.99, top=0.9, wspace=0.1)

    conf = all_conf
    sets = all_sets
    bottom = np.zeros(x.shape)
    for u, (ax, s) in enumerate(zip(axs.ravel(), ["test"] + ADV_SETS)):
        conf_s = conf[sets == s]
        hist, bin_edges = np.histogram(conf_s, bins=bins)
        hist = hist / len(conf)
        kwargs = {"facecolor": cmap(u)}
        label = s
        if s == "test":
            label = "test set"
            kwargs["facecolor"] = "gray"
        kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(conf_s.reshape(-1, 1))
        #y = kde.score_samples(x.reshape(-1, 1))
        y = np.exp(kde.score_samples(x.reshape(-1, 1)))
        #ax.bar(bins[:-1], hist, width=0.05, align="edge", bottom=bottom,
        #       label=s, hatch=SET_HATCH[s],
        #       linewidth=0.5, edgecolor="black", **kwargs)
        #ax.plot(bins[:-1], hist+bottom, label=s)
        ax.fill_between(x, bottom, bottom+y, label=label, edgecolor="black",
                        linewidth=0.5, **kwargs)
        #print(f"{s:3}", hist)
        #bottom += y

        #ax.set_title(s)
        ax.set_xlabel("Confidence")
        #ax.set_ylabel("Density")
        ax.set_xlim([0.5, 1])
        ax.set_ylim([0.0, ax.get_ylim()[1]])
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels([".5", ".75", "1"])
        #ax.set_yticklabels([""] * len(ax.get_yticks()))
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.legend(loc="upper center")


    axs[0].set_ylabel("Gaussian\nkernel density")
    #for k in [0, 1, 2]: plt.setp(axs[k].get_xticklabels(), visible=False)

    if SAVEFIGS:
        figname = f"conf_density_{model_type}"
        fig.savefig(os.path.join("figures", f"{figname}.svg"))
        subprocess.run(["/home/laurens/repos/dotfiles/scripts/svg2latex", f"{figname}.svg"], cwd="figures")
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf_tex"),
                        os.path.join(FIG_DIR, f"{figname}.pdf_tex"))
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf"),
                        os.path.join(FIG_DIR, f"{figname}.pdf"))

    plt.show()

def plot_aucs(aucs):
    for dname, adv_sets in aucs.items():
        fig, ax = plt.subplots()
        ax.set_title(dname)
        for k, detect_alg in enumerate(DETECT_ALGS):
            x = np.arange(len(adv_sets)) + k/(len(DETECT_ALGS) + 1)
            y = [adv_sets[s][detect_alg]["mean"] for s in adv_sets.keys()]
            e = [adv_sets[s][detect_alg]["std"] for s in adv_sets.keys()]
            ax.bar(x, y, yerr=e, width=0.6/(len(DETECT_ALGS)+1), label=detect_alg)
        ax.set_xticks(np.arange(len(ADV_SETS)) + 0.3)
        ax.set_xticklabels(ADV_SETS)
        ax.legend()
    plt.show()

def tabulate_aucs(aucs):
    idx = pd.MultiIndex.from_product([DETECT_ALGS, ADV_SETS], names=["Algorithm", "Set"])
    df_aucs = pd.DataFrame("-", index=idx, columns=THE_DATASETS)

    for dname, adv_sets in aucs.items():
        for adv_set in adv_sets.keys():
            for k, detect_alg in enumerate(DETECT_ALGS):
                df_aucs.loc[(detect_alg, adv_set), dname] =\
                        adv_sets[adv_set][detect_alg]["mean"]

    return df_aucs

def plot_aucs_per_conf(all_aucs_per_conf):
    fig, axs = plt.subplots(1, len(all_aucs_per_conf), sharey=False, figsize=(20, 5))
    fig.subplots_adjust(left=0.05, right=0.95)

    for ax, (dataset, aucs_per_conf) in zip(axs, all_aucs_per_conf.items()):
        x = 0.5 + np.array([x[0] for x in CONF_BUCKETS])/2
        shift = 0.25 / len(CONF_BUCKETS)

        ls = ["o", "x", "v", "^"]
        Ns = 0
        Nadvs = 0

        for i, (detect_alg, vs) in enumerate(aucs_per_conf.items()):
            y = [u["mean"] for u in vs]
            e = [u["std"] for u in vs]
            Ns = [u["N"] for u in vs]
            Nadvs = [u["Nadv"] for u in vs]

            ax.plot(x+shift, y, "-", marker=ls[i], label=detect_alg)
            #ax.errorbar(x+shift, y, yerr=e, marker=ls[i], label=detect_alg)

        ax.set_xticks(list(x)+[1.0])
        ax.set_xlabel("Confidence")
        ax.set_ylabel("AUC")
        fig.suptitle("AUC for example with confidence in bucket (in gray: how many such examples are there?)")
        ax.set_ylim([0.4, 1.050])
        ax.set_title(dataset)

        for x, n in zip(x, Ns):
            ax.text(x+shift, 1.03, str(n), ha="center", va="top", c="gray")

    axs[0].legend()

    plt.show()

def plot_accs(all_accs):
    fig, axs = plt.subplots(1, len(all_accs), sharey=False, figsize=(20, 5))
    fig.subplots_adjust(left=0.05, right=0.95)

    for ax, (dataset, accs) in zip(axs, all_accs.items()):
        for i, (detect_alg, vs) in enumerate(accs.items()):
            #print(dataset, detect_alg, vs.keys())
            x = vs["conf"]
            y = vs["acc_mean"]
            e = vs["acc_std"]
            a = auc(x, y) * 2.0
            print("auc", dataset, detect_alg, a)

            if detect_alg in DETECT_ALGS:
                l, = ax.plot(x, y, "-", label=detect_alg)
                ax.fill_between(x, y-e/2, y+e/2, alpha=0.2, fc=l.get_color())
                #ax.errorbar(x, y, yerr=e, marker=ls[i], label=detect_alg)
            elif detect_alg == "nadvs":
                ax.plot(x, y, "-", lw=0.5, color="gray", label="Fraction adversarial")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(dataset)


        ax.set_xlim([0.5, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.text(0.50, -0.1, "Least confident", ha="left")
        ax.text(1.05, -0.1, "Most confident", ha="right")

    fig.suptitle("Sliding window accuracy with fixed threshold set to adv vs. normal ratio")
    axs[0].legend()
    plt.show()


def plot_vary_refset_size(r, model_type):
    markers = ["o", "x", "D", "v", "^", "s", "h", "H"]
    lstyles = list(ALG_LS.values())
    cmap = cm.get_cmap("tab10")
    fig, axs = plt.subplots(1, 2, figsize=(3.2, 1.5), num=f"subset")
    fig.subplots_adjust(left=0.15, bottom=0.22, right=0.96, top=0.82, wspace=0.45, hspace=0.60)

    for i, dname in enumerate(["covtype", "mnist2v4", "ijcnn1"]):
        data = r[dname]
        subsets = data[0]["subsets"]
        for j, k in [(0, "aucs"), (1, "times")]:
            if k == "times":
                a = np.vstack([np.array(data[fold][k]) / data[fold][k][-1] for fold in range(len(data))])
            else:
                a = np.vstack([data[fold][k] for fold in range(len(data))])
            y = a.mean(axis=0)
            e = a.std(axis=0)
            axs[j].errorbar(subsets, y, yerr=e, fmt=lstyles[i],
                    marker=markers[i],
                    color=cmap(i), markersize=3, label=dname)
            axs[j].set_xlabel("subset size")
            axs[j].set_xlim([0.0, 1.1])
            axs[j].set_xticks(subsets)
            xlabels = [f"{x}".lstrip("0") for x in subsets]
            xlabels[-1] = "1"
            axs[j].set_xticklabels(xlabels)

    #axs[0].legend()
    handles, labels = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    axs[0].set_ylim([0.5, 1.01])
    axs[0].set_ylabel("ROC AUC")
    axs[1].set_ylabel("time fraction")
    #axs[1].set_yscale("log")

    #d = .04
    #axs[0].plot(
    #        (0, d, -d, d, 0),
    #        (0, d, 3*d, 5*d, 6*d),
    #        transform=axs[0].transAxes, color="k", clip_on=False)

    #kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    #ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    if SAVEFIGS:
        figname = f"vary_refset_size_{model_type}"
        fig.savefig(os.path.join("figures", f"{figname}.svg"))
        subprocess.run(["/home/laurens/repos/dotfiles/scripts/svg2latex", f"{figname}.svg"], cwd="figures")
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf_tex"),
                        os.path.join(FIG_DIR, f"{figname}.pdf_tex"))
        shutil.copyfile(os.path.join("figures", f"{figname}.pdf"),
                        os.path.join(FIG_DIR, f"{figname}.pdf"))

    plt.show()


@click.command()
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "groot"]), default="xgb")
@click.option("-N", "N", default=100)
@click.option("--ratio", default=5)
@click.option("--nfolds", default=NFOLDS)
@click.option("--cache_dir", default="cache", show_default=True)
@click.option("--seed", default=SEED)
def analyze(model_type, N, ratio, nfolds, cache_dir, seed):
    #aucs = collect_aucs(model_type, N, ratio, nfolds, cache_dir, seed)
    ##plot_aucs(aucs)
    #df_aucs = tabulate_aucs(aucs)
    #print(df_aucs)

    #print("Averaged over adversarial sets")
    #print("-- mean -- ")
    #print(df_aucs.groupby(level=[0]).mean().T)
    #print("-- std -- ")
    #print(df_aucs.groupby(level=[0]).std().T)
    #print("Averaged over all datasets, per adversarial set")
    #print(df_aucs.mean(axis=1))
    #print("Averaged over all datasets and adversarial sets")
    #print(df_aucs.groupby(level=[0]).mean().mean(axis=1))

    #all_aucs_per_conf = collect_aucs_per_conf(model_type, N, ratio, nfolds, cache_dir, seed)
    #plot_aucs_per_conf(all_aucs_per_conf)

    #all_accs = collect_accs_fixed_threshold(model_type, N, ratio, nfolds, cache_dir, seed)
    #plot_accs(all_accs)


    ###########


    global SAVEFIGS
    SAVEFIGS=True

    #dnames = ["phoneme"]
    dnames = ["phoneme", "covtype", "mnist2v4", "ijcnn1", "webspam", "calhouse", "fmnist2v4"]
    #dnames = ["covtype", "mnist2v4", "ijcnn1", "calhouse"]
    #dnames = ["mnist2v4", "fmnist2v4"]

    per_set, per_alg, per_set_alg, per_confdelta = collect_results(dnames,
                                                                   model_type,
                                                                   N, ratio,
                                                                   nfolds,
                                                                   cache_dir,
                                                                   seed)
    display_results(per_set, per_alg, per_set_alg)
    #plot_confdelta(per_confdelta, per_alg)
    #plot_confdist2(per_confdelta, model_type)


    ###########

    
    # python analyze_results.py -N 500 --cache_dir=cache_pinacs --ratio 4
    #plot_vary_refset_size(load(os.path.join(cache_dir, f"vary_refszet_size_{model_type}.joblib")), model_type)

    

if __name__ == "__main__":
    analyze()
