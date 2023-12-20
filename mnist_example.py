# based on: https://github.com/laudv/veritas/blob/finiteprec/examples/mnist.ipynb

import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import ocscore

X_mc, y_mc = datasets.fetch_openml(data_id=554, return_X_y=True, as_frame=False)
y_mc = y_mc.astype(int)

class0, class1 = 1, 7
mask = (y_mc == class0) | (y_mc == class1)
X = X_mc[mask, :]
y = (y_mc[mask] == class1)

xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=13)

params = {
    "n_estimators": 50,
    "eval_metric": "error",
    
    "tree_method": "hist",
    "seed": 135,
    "max_depth": 5,
    "learning_rate": 0.4
}
model = xgb.XGBClassifier(**params)

t = time.time()
model.fit(X, y)
print(f"XGB trained in {time.time()-t} seconds")

ytrain_pred = model.predict(xtrain)
ytest_pred = model.predict(xtest)
acc_train = accuracy_score(ytrain, ytrain_pred)
acc_test = accuracy_score(ytest, ytest_pred)

print(f"Train acc: {acc_train:.3f}, test acc: {acc_test:.3f} wrt true labels")




# Find a reference set, e.g. set of correctly classified training examples
xref_mask = ytrain_pred == ytrain
xref = xtrain[xref_mask]
yref = ytrain[xref_mask]
xref0 = xref[yref==0]
xref1 = xref[yref==1]

# Find some examples to compute the OC-score of
# You could add adversarial examples here to see how OC-score does
# See this example on how to generate adversarial examples with Veritas:
#   https://github.com/laudv/veritas/blob/finiteprec/examples/mnist.ipynb
rng = np.random.default_rng(seed=18)
query_indices = rng.choice(xtest.shape[0], 100)
xquery = xtest[query_indices]
yquery_pred = ytest_pred[query_indices]

# Add random noise to one, and see if they come up
xquery[0, :] += 50.0*rng.random(xquery.shape[1])

# Add in an eight, and see if it shows up
xquery[1, :] = X_mc[np.argmax(y_mc==8), :]

# To compute the OC-score, we need the OCs of the refset and the test set
# examples (i.e., an identifier for each reached leaf). We use the scikit-learn
# apply function for this here. In the experiments, we used ocscore.mapids to
# ensure that the identifiers are in 0..255.
# We are used max-depth 4 here, so the leaf id is always < 255.
dtype = np.uint16
idref0 = model.apply(xref0).astype(dtype)
idref1 = model.apply(xref1).astype(dtype)
idquery = model.apply(xquery).astype(dtype)


### WITH VERITAS ##
#import veritas
#import ocscore_veritas
#at = veritas.get_addtree(model)
#idref0 = ocscore_veritas.mapids(at, xref0, dtype)
#idref1 = ocscore_veritas.mapids(at, xref1, dtype)
#idquery = ocscore_veritas.mapids(at, xquery, dtype)


# Compute OC-score with respect to refset examples with the same (predicted) label
t = time.time()
S0 = ocscore.ocscores(idref0, idquery[yquery_pred==0])
S1 = ocscore.ocscores(idref1, idquery[yquery_pred==1])
S = np.zeros(idquery.shape[0], dtype=int)
S[yquery_pred==0] = S0
S[yquery_pred==1] = S1
print(f"OC-scores in {1000.0*(time.time() - t):.1f} ms")

print(S0)
print(S1)

print("OC-scores:")
print(S)

normalest = np.argsort(S)
weirdest = normalest[::-1]

fig, axs = plt.subplots(2, 5, figsize=(12, 5))
kwargs = {"cmap": "binary", "vmin": 0, "vmax": 255, "interpolation": "none"}
fig.suptitle("Top: most normal, Bottom: most abnormal")
for ax, i in zip(axs[0, :], normalest):
    ax.imshow(xquery[i].reshape((28, 28)), **kwargs)
    ax.set_title(f"OC-score {S[i]}")
for ax, i in zip(axs[1, :], weirdest):
    ax.imshow(xquery[i].reshape((28, 28)), **kwargs)
    ax.set_title(f"OC-score {S[i]}")
plt.show()



