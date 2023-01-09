import data_and_trees

THE_DATASETS = {
    "phoneme": "Phoneme_20-6-50",
    "spambase": "Spambase_20-5-50",
    "covtype": "CovtypeNormalized_80-6-50",
    "higgs": "Higgs_100-8-10",
    "ijcnn1": "Ijcnn1_50-5-90",
    "mnist2v4": "MnistBinClass[2,4]_50-5-70",
    "fmnist2v4": "FashionMnistBinClass[2,4]_50-5-90",
    "webspam": "Webspam_50-5-90",
    "calhouse": "CalhouseClf_100-5-50",
}

def parse_dataset0(value):
    # parameterized dataset
    try:
        i0 = value.index('[')
        i1 = value.index(']')
    except:
        return getattr(data_and_trees, value)()
    dataset_name = value[0:i0]
    params = list(map(int, value[i0+1:i1].split(',')))
    #print("PARAMETERIZED", dataset_name, params)
    return getattr(data_and_trees, dataset_name)(*params)

# <DATASETNAME>_<num_trees>-<tree_depth>-lr<learning_rate*100>
# or <key> in THE_DATASETS
def parse_dataset(value):
    if value in THE_DATASETS:
        return parse_dataset(THE_DATASETS[value])
    else:
        print(f"No parameters for {value}, parsing...")
        i0 = value.rindex("_")
        i2 = value.rindex("-")
        i1 = value.rindex("-", 0, i2-1)
        if i0 == -1 or i1 == -1 or i2 == -1:
            raise ValueError()
        dataset_name = value[0:i0]
        num_trees = value[i0+1:i1]
        tree_depth = value[i1+1:i2]
        learning_rate = value[i2+1:]
        #print("PARAMS", dataset_name, num_trees, tree_depth, learning_rate)
        d = parse_dataset0(dataset_name)
        return d, int(num_trees), int(tree_depth), float(int(learning_rate)/100.0)

def get_dataset(ctx, param, value):
    return parse_dataset(value)
