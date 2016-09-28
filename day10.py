import pandas
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from matplotlib import pylab
import numpy
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
# load dataset

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv("/Users/loveltyoic/Downloads/pima-indians-diabetes.data.txt", names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare configuration for cross validation test harness
num_folds = 10
num_instances = len(X)
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)

# models.append(('Ridge', grid.best_estimator_))
# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'
def plot_roc(auc_score, name, tpr, fpr, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.fill_between(fpr, tpr, alpha=0.5)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve (AUC = %0.2f) / %s' % (auc_score, label), verticalalignment="bottom")
    pylab.legend(loc="lower right")
    filename = name.replace(" ", "_")
    pylab.show()
    # pylab.savefig(os.path.join(CHART_DIR, "roc_" + filename + ".png"), bbox_inches="tight")

for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.33, random_state=24)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 0]
    print y_test
    print y_pred
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr, reorder=True)
    # plot_roc(auc_score, 'roc curve', fpr, tpr)
    print roc_thresholds
    print auc_score


# print(grid.best_score_)
# print(grid.best_estimator_.alpha)

# random = RandomizedSearchCV(estimator=model, param_distributions=param_grid)
# random.fit(X,Y)
# print random.best_score_
# print random.best_estimator_.alpha