import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from util import *
from datasets import THE_DATASETS, parse_dataset

from pprint import pprint
from sklearn.metrics import roc_auc_score, auc

DETECT_ALGS = ["ocscore", "ambig", "iforest", "mlloo"]
ADV_SETS = ["lt", "kan", "ver", "cub"]

CONF_BUCKETS = np.linspace(0.0, 1.0, 4)
CONF_BUCKETS[-1] = 1.0001 # make last conf_hi inclusive
CONF_BUCKETS = list(zip(CONF_BUCKETS, CONF_BUCKETS[1:]))

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
        aucs[dname] = collect_aucs_for_dataset(dataset,
                                                            model_type, N,
                                                            ratio, nfolds,
                                                            cache_dir, seed)
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

    acc_grid = {detect_alg: {} for detect_alg in DETECT_ALGS}
    acc_grid["n"] = {}

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
                if detect_alg not in Sall:
                    Sall[detect_alg] = S
                else:
                    Sall[detect_alg] = np.hstack((Sall[detect_alg], S[Nsample:]))

        assert len(ydetect) == Xall.shape[0]
        for detect_alg in DETECT_ALGS:
            assert len(Sall[detect_alg]) == Xall.shape[0]
        delta = np.sum(np.abs(Xall[Nsample:, :]-Xbase_all), axis=1)
        delta_buckets = np.linspace(min(delta), max(delta), len(CONF_BUCKETS)+2)[1:]
        delta_buckets[-1] += 0.001
        delta_buckets = list(zip(delta_buckets, delta_buckets[1:]))

        pred_prob = at.predict_proba(Xall)
        conf = np.abs(pred_prob - 0.5) + 0.5
        conf_perm = np.argsort(conf)

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

            # grid delta and conf
            if "values" not in acc_grid[detect_alg]:
                acc_grid[detect_alg]["values"] = np.zeros((nfolds,
                            len(CONF_BUCKETS), len(delta_buckets)))
            conf_adv = conf[Nsample:]
            ydetect_acc_adv = conf[Nsample:]
            for i, (conf_lo, conf_hi) in enumerate(CONF_BUCKETS):
                for j, (delt_lo, delt_hi) in enumerate(delta_buckets):
                    index = (conf_lo <= conf_adv)\
                          & (conf_adv < conf_hi)\
                          & (delt_lo <= delta)\
                          & (delta < delt_hi)
                    bucket = ydetect_acc_adv[index]
                    print("grid", detect_alg, conf_lo, delt_lo, len(bucket), np.mean(bucket))
                    acc_grid[detect_alg]["values"][fold, i, j] = np.mean(bucket)

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

    print(acc_grid)

    return accs


def collect_accs_fixed_threshold(model_type, N, ratio, nfolds, cache_dir, seed):
    accs = {}
    #for dname in THE_DATASETS:
    for dname in ["covtype", "spambase", "mnist2v4", "calhouse", "ijcnn1"]:
        dataset = parse_dataset(dname)
        accs[dname] = collect_accs_fixed_threshold_for_dataset(dataset,
                                                               model_type, N,
                                                               ratio, nfolds,
                                                               cache_dir, seed)
    return accs

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

    all_accs = collect_accs_fixed_threshold(model_type, N, ratio, nfolds, cache_dir, seed)
    plot_accs(all_accs)
    

if __name__ == "__main__":
    #configure_matplotlib()
    analyze()
