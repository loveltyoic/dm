import xgboost as xgb
import pdb
import numpy as np
# read in data
from sklearn.cross_validation import train_test_split
import random
#dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
#dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
#param = {'nthread': 8, 'num_class':9, 'objective':'multi:softprob', "eval_metric":"mlogloss" }
param = {'nthread': 8,
         "objective": "binary:logitraw",
         "eval_metric": "auc", "bst:eta":  0.1,
         "bst:max_depth": 4,
         "min_child_weight": 4,
         "gamma": 0.1,
         "lambda": 500,
         "scale_pos_weight": 6000/8000.0,
         "eval_matric": "ams@0.15",
         "subsample": 0.8,
         'early_stopping_rounds':100}
num_round = 1000
#bst = xgb.train(param, dtrain, num_round)
# make prediction
#preds = bst.predict(dtest)


threshold_ratio = 0.15
# load in training data, directly use numpy
data = np.loadtxt('/Users/loveltyoic/Downloads/tousu.csv', delimiter=',', skiprows=1, dtype='float' )
#dtest = np.loadtxt('/home/zhli7/test.csv', delimiter=',', skiprows=1, dtype='float')
train,test = train_test_split(data, test_size = 0.5,random_state=1225)
train,val = train_test_split(train, test_size = 0.2,random_state=1225)
false_ix = (train[:, 0] == 0)
false_data = train[false_ix]
false_amplified = []
#pdb.set_trace()
for i in range(sum(train[:, 0] == 1)):
    false_amplified.append(random.choice(false_data))

dtest = test[:, 1:]
#dtrain = data[0:8000, 1:]
dtrain = np.concatenate((np.array(false_amplified), train[train[:, 0] == 1]), axis=0)
#label = np.rint(data[0:8000,0])
label = dtrain[:, 0]
# weight = np.array([10 if l == 0 else 1 for l in label])
#pdb.set_trace()
print ('finish loading from csv ')
xgb_train = xgb.DMatrix(dtrain[:, 1:], missing = -999.0, label= label)#, weight = weight )
xgb_test = xgb.DMatrix(dtest, missing = -999.0)
# xgb_cross = xgb.DMatrix(dtrain, missing = -999.0)
xgb_val = xgb.DMatrix(val[:, 1:], label=val[:, 0])
watchlist = [ (xgb_val,'val'), (xgb_train, 'train') ]
# cv = xgb.cv(param, xgb_train, nfold=3, seed = 0)
#pdb.set_trace()
bst = xgb.train(param, xgb_train, num_round, evals=watchlist)
ypred = bst.predict(xgb_test )
# ycross = bst.predict(xgb_cross)
#pdb.set_trace()
pred = np.array([1 if i > 0.5 else 0 for i in ypred])
test_label = test[:, 0]
#test = label
tn = np.sum(np.logical_and(pred == 0, test_label == 0))
tp = np.sum(np.logical_and(pred == 1, test_label == 1))
fp = np.sum(np.logical_and(pred == 1, test_label == 0))
fn = np.sum(np.logical_and(pred == 0, test_label == 1))
print float(tp) / (tp + fp)
print float(tp) / (tp + fn)
print float(tn) / (tn + fn)
print float(tn) / (tn + fp)

