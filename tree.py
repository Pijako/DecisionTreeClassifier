#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import random
import pandas as pd
import matplotlib
import numpy as np
import os
import subprocess

import matplotlib.pyplot as plt

class Model:

	def __init__(self, tree):
		self.tree = tree
		self.accuracy = 0
		self.precision = 0
		self.recall = 0
		self.n_node = 0
		self.vp = 0
		self.fp = 0
		self.vn = 0
		self.fn = 0
		self.node_id=[]


	def singlePredict(self, s):
		node = self.tree

		while(node['type']==0):
			if(s[node['feature']] <= node['threshold']):
				node = node['left']
			else:
				node = node['right']

		print("Class : "+str(s[-1])+" Predicted : "+str(node['Class']))
		if((s[-1] == 1) & (node['Class']==1)):
			self.vp = self.vp + 1
		if((s[-1] == 0) & (node['Class']==0)):
			self.vn = self.vn + 1	
		if((s[-1] == 1) & (node['Class']==0)):
			self.fp = self.fp + 1	
		if((s[-1] == 0) & (node['Class']==1)):
			self.fn = self.fn + 1		

		return [s[0], s[1], node['Class']]

	def predict(self, sk):

		sk = np.array(sk).transpose().tolist()

		return list(map(self.singlePredict, sk))

	def setMetrics(self):

		if((float(self.vp)+float(self.vn))!=0):
			self.accuracy = (float(self.vp)+float(self.vn))/(float(self.fn)+float(self.fp)+float(self.vp)+float(self.vn))
		else:
			self.accuracy = 0.0

		if((float(self.vp)+float(self.fp))!=0):
			self.precision = float(self.vp)/(float(self.vp)+float(self.fp))
		else:
			self.precision = 0.0

		if((float(self.vp)+float(self.fn))!=0):	
			self.recall = float(self.vp)/(float(self.vp)+float(self.fn))
		else:
			self.recall = 0.0

	def printConfusionMatrix(self):

		print("Accuracy : "+str(self.accuracy))
		print("Precision : "+str(self.precision))
		print("Recall : "+str(self.recall))

	def exportGraph(self, leaf):

		previous = self.n_node

		if(leaf['type'] == 0):
			label_previous = "feature "+str(leaf['feature'])+" <= "+str(leaf['threshold'])
			self.node_id.append(self.n_node)
			graph.write("node"+str(self.n_node)+' [label="'+str(label_previous)+'"];\n')

			if(leaf['left']['type']==0):
				while(self.n_node in self.node_id):
					self.n_node = self.n_node+1
				self.node_id.append(self.n_node)	
				label_left = "feature "+str(leaf['left']['feature'])+" <= "+str(leaf['left']['threshold'])
				graph.write("node"+str(self.n_node)+' [label="'+str(label_left)+'"];\n')
				graph.write("node"+str(previous)+" -> node"+str(self.n_node)+"[label=\"True\"];\n")
				self.exportGraph(leaf['left'])
				
			else:
				label = int(leaf['left']['data'][-1][0])
				label_left = "Class"+str(label)
				while(self.n_node in self.node_id):
					self.n_node = self.n_node+1
				self.node_id.append(self.n_node)
				graph.write("node"+str(self.n_node)+' [label="'+str(label_left)+'"];\n')
				graph.write("node"+str(previous)+" -> node"+str(self.n_node)+"[label=\"True\"];\n")

			if(leaf['right']['type']==0):
				while(self.n_node in self.node_id):
					self.n_node = self.n_node+1
				self.node_id.append(self.n_node)	
				label_right = "feature "+str(leaf['right']['feature'])+" <= "+str(leaf['right']['threshold'])
				graph.write("node"+str(self.n_node)+' [label="'+str(label_right)+'"];\n')
				graph.write("node"+str(previous)+" -> node"+str(self.n_node)+"[label=\"False\"];\n")
				self.exportGraph(leaf['right'])
		
			else:
				label_right = "Class"+str(int(leaf['right']['data'][-1][0]))
				while(self.n_node in self.node_id):
					self.n_node = self.n_node+1
				self.node_id.append(self.n_node)
				graph.write("node"+str(self.n_node)+' [label="'+str(label_right)+'"];\n')
				graph.write("node"+str(previous)+" -> node"+str(self.n_node)+"[label=\"False\"];\n")
		
		
		else:
			label = int(leaf['right']['data'][-1][0])
			label_previous = "Class"+str(label)
			while(self.n_node in self.node_id):
				self.n_node = self.n_node+1
			self.node_id.append(self.n_node)
			graph.write("node"+str(self.n_node)+' [label="'+label_previous+'"];\n')


	def getTree(self):
		self.exportGraph(self.tree)


class DecisionTreeClassifier:

	def __init__(self):
		pass

	def fit(self, sk):

		return Model(self.train(sk))


	def train(self, sk):

		g = self.computeGini(sk[2])
		if(g==0):
			print("--> Gini = 0. Sous-ensemble classé")
			final = int(sk[2][0])

			result = {'type':1, 'Class':final, 'data':sk}

			return result
			
		else:
			print("--> Gini = "+str(g)+" : On classe ce sous-ensemble")
			n_sample = np.transpose(np.array(sk)).size
			f, t, g = self.best_feature_and_threshold(sk)
			sa, sb = self.discriminate(sk, f, t)

			left = self.train(sa)
			right = self.train(sb)

			tree = {'type':0, 'feature':f, 'threshold':t, 'gini':g, 
					  'label':"feature "+str(f)+" <= "+str(t),'left':left, 'right':right}

		return tree


	def if_classified(self, sk):
		return self.computeGini(sk) == 0

	def computeGini(self, labels):
		pk = self.computePk(labels)

		return round(1-sum([pow(p, 2) for p in pk]), 3)

	def computeAverageGini(self, sasb):
		sa = sasb[0]
		sb = sasb[1]

		card = len(sa[2])+len(sb[2])
		gini1 = self.computeGini(sa[2])*len(sa[2])/card
		gini2 = self.computeGini(sb[2])*len(sb[2])/card

		return round(gini1+gini2,3)


	def computePk(self, labels):
		n = len(labels)
		
		card_a = sum(labels)
		card_b = n - card_a
		
		pk = [float(card_a)/n, float(card_b)/n]

		return np.array(pk)

	def best_feature_and_threshold(self, sk):

		final_feature = 0
		final_threashold = 0
		final_gini = 1
		n_features = len(sk.tolist())
		card = len(list(sk[0]))

		print("Test parmis "+str(n_features-1)+" features")

		for f in range(0, n_features-1):
		
			thresholds = [round((sk[f][i+1]+sk[f][i])/2, 3) for i in range(0, len(sk[f])-1)]
			average_ginis = list(map(lambda t : self.computeAverageGini(self.discriminate(sk, f, t)), thresholds))

			best_gini_feature = min(average_ginis)
			t = thresholds[average_ginis.index(best_gini_feature)]

			if( best_gini_feature < final_gini ):
				final_feature = f
				final_threashold = t
				final_gini = best_gini_feature
				
		return final_feature, final_threashold, final_gini

	def discriminate(self, sk, f, t):
		
		sk = np.transpose(np.array(sk))
		sk = list(sk)
		sa = [list(s) for s in sk if (s[f] <= t)]

		sb = [list(s) for s in sk if (list(s) not in sa)]
		sa = np.transpose(np.array(sa))
		sb = np.transpose(np.array(sb))

		return [sa, sb]


print("--> Read or generate data ..")
data = pd.read_csv('data.csv', encoding='utf8', names=['x', 'y', 'label'])
data = np.transpose(np.array(data))
graph = open("graph.dot", "w")

colors=['red', 'blue']
data2 = []
a = [2, 2]
b = [-2, -2]
c = [-1, 3]
d = [2, 0]
for i in range(0,20):
	data2.append([a[0]+random.uniform(-1, 1), a[1]+random.uniform(-1, 1), 1])
	data2.append([b[0]+random.uniform(-1, 1), b[1]+random.uniform(-1, 1), 1])
	data2.append([c[0]+random.uniform(-1, 1), c[1]+random.uniform(-1, 1), 0])
	data2.append([d[0]+random.uniform(-1, 1), d[1]+random.uniform(-1, 1), 0])

data2_train = data2[0:int(len(data2)*0.8)]
data2_test = np.array([d for d in data2 if d not in data2_train])
print("train : "+str(len(data2_train)))
print("test : "+str(len(data2_test)))

data2_train = np.transpose(np.array(data2_train))
data2_test = np.transpose(np.array(data2_test))

graph.write("digraph tree {\n")

dtc = DecisionTreeClassifier()
print("--> Training ..")
model = dtc.fit(data2_train)
model.getTree()

print("--> Predicting..")
data2_predictions = model.predict(data2_test)
model.setMetrics()
model.printConfusionMatrix()

graph.write("}\n")
graph.close()

data2_predictions = np.array(data2_predictions).transpose()
plt.scatter(data2_train[0], data2_train[1], c=data2_train[2], marker='o', cmap=matplotlib.colors.ListedColormap(colors))
#plt.scatter(data2_test[0], data2_test[1], c=data2_test[2], marker='x', cmap=matplotlib.colors.ListedColormap(colors))
plt.scatter(data2_predictions[0], data2_predictions[1], c=data2_predictions[2], marker='x', cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
subprocess.run(["xdot", "graph.dot"])

