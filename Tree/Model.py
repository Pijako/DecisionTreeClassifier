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

		print(str(s[-1])+" et "+str(node['Class']))
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