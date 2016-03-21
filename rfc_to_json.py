'''
installation instructions for python
https://store.enthought.com/downloads/#default
once you have python installed, you'll want install the following modules:
pip install pandas, numpy, sklearn, json

rfc_to_json 
	1. fake up some random categorical data
	2. train an instance of sklearn's RandomForestClassifier to label the faked training data
	3. write out the trees in a sklearn RandomForestClassifier to json files (one tree per json file)
	4. import the json trees and generate predictions for a new 


#fake up some training and test data with five features and two labels
df = fake_data()
#train sklearn RandomForestClassifier
clf = train_classifier(df)

#export json files with tree_to_json
export_RFC_to_json(clf, list(df.columns.values[0:-1]))
#import forests from json files
forest = import_RFC()

#select a sample data point
x = df.iloc[0, 0:5]
x = pd.Series({'feature0': 0.9, 'feature1': 0.9,'feature2': 0.9,'feature3': 0.9,'feature4': 0.9})
prediction = pred_from_json(forest, x)

#make predictions for the test set (note that sklearn expects a numpy array and pred_from_json expects a dictionary object)
df_test = fake_data()
predicted_from_json = pred_from_json(forest, df_test.iloc[0, 0:5])
predicted_from_clf = clf.predict_proba(df_test.iloc[0, 0:5].values)

#export png images of the trees
write_tree_image(clf):


exercises:
1. change the number of features or labels in the fake data
2. what happens if the labels can be 0, 1 or 2 instead of just 0 or 1?
3. tweak pred_from_json to generate output in the same format as clf.predict_proba
'''


import pandas as pd #pandas lets store your data in a nice database-ish structure
import numpy as np
import os
import json
from sklearn import tree

import tree_to_json

#comment these libraries out if you can't install them, they're only needed for exporting plots of the trees
import pydot
import StringIO
from IPython.core.display import Image



'''
fake up some data
'''
def make_label(x):
	if x[0] > 0.7 and x[1] > 0.7:
		if np.random.random(1)[0] < 0.85:
			return 1
		else:
			return 0	
	if x[2] > 0.7 and x[3] > 0.7:	
		if np.random.random(1)[0] < 0.85:
			return 1
		else:
			return 0
	
	return 0	

#df = fake_data()
#df_test = fake_data()
def fake_data():
	features = 5
	n = 100
	df = pd.DataFrame(np.random.random(n*features).reshape((n, features)), columns= ['feature'+str(i) for i in range(features)])

	df['label0'] = 0
	df['label0'] = df.iloc[:, 0:features].apply(make_label, axis=1)
	df['label1'] = 0
	df['label1'] = df.iloc[:, 0:features].apply(make_label, axis=1)	

	return df


#clf = train_classifier(df)
def train_classifier(df):
	x_train = df.iloc[:, 0:df.shape[1]-2]
	y_train = df[['label0', 'label1']]

	clf = ensemble.RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)  #, class_weight={0:0.05, 1:0.95}
  
	clf.fit(x_train.values, y_train.values)
	print clf.predict(x_train.values)[0:10], y_train.values[0:10]
	return clf



#export_RFC_to_json(clf, list(df.columns.values[0:-1]))
def export_RFC_to_json(clf, feature_names):
  for i in xrange(clf.n_estimators):

    jsontree = tree_to_json.treeToJson(clf.estimators_[i].tree_, feature_names)

    treedir  = 'jsontrees/'
    try:
    	os.stat(treedir)
    except:
    	os.mkdir(treedir) 

    text_file = open(treedir+'tree'+str(i)+'.txt', 'w')

    text_file.write(jsontree)

    text_file.close()		


#forest = import_RFC()
def import_RFC():
	treedir  = 'jsontrees/'
	forest = []

	i = 0

	while os.path.isfile(treedir+'tree'+str(i)+'.txt'):
		with open(treedir+'tree'+str(i)+'.txt') as json_file:
			forest  += [json.load(json_file)]
		i += 1	

	return forest




'''
traverse tree, evaluating the decision rule at each node for the values in x
'''
def pred_from_tree(t, x):
	try: 
		#the node is not a leaf
		f, rule, threshold = t['rule'].split()
		if eval( str( str(x[f]) +rule+threshold ) ) : #choose left child
			return pred_from_tree(t['left'], x)
		else:
			return pred_from_tree(t['right'], x)	
	except:		
		#the node is a leaf
		r = []
		for i in eval(t['value']):
			r += [1.0*i[1]/(i[0]+i[1])]
			
		#print r	
		return np.array(r)


'''
make a prediction for a single row
'''
def pred_from_json(forest, x):
	
	for i in xrange(len(forest)):
		if i == 0:
			p = pred_from_tree(forest[i], x)
		else:
			p += pred_from_tree(forest[i], x)


	return 1.0*p/len(forest)		




def write_tree_image(clf):
	for i in xrange(clf.n_estimators):
		dot_data = StringIO.StringIO()
		tree.export_graphviz(clf.estimators_[0], out_file=dot_data)
		graph = pydot.graph_from_dot_data(dot_data.getvalue())
		image = graph.write_png("tree"+str(i)+".png")









