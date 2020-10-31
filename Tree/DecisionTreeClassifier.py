import math

class DecisionTreeClassifier:

	def __init__(self, max_depth=6, metrics="gini"):
		pass
		self.max_depth = max_depth
		self.metrics = metrics
		self.depth = 1

	def fit(self, sk):

		return Model(self.train(sk))


	def train(self, sk):

		g = self.computeGini(sk[2])
		if(g==0 | self.depth >= self.max_depth):
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

		  self.depth = self.depth + 1

		return tree


	def if_classified(self, sk):
		return self.computeGini(sk) == 0

	def computeEntropy(self, labels):
		pk = self.computePk(labels)

		return -sum([p*math.log(p) for p in pk])

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