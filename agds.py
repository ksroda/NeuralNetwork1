import csv

def importIrises():
	iris = []
	with open('iris.csv', 'rb') as csvfile:
		iris_dataset = csv.reader(csvfile, delimiter=',')
		for row in iris_dataset:
			_str = []
			for i in row:
				try:
					float(i)
					_str.append(float(i))
				except ValueError:
					if i == 'Iris-setosa':
						_str.append('1.0')
					elif i == 'Iris-versicolor':
						_str.append('2.0')
					elif i == 'Iris-virginica':
						_str.append('3.0')
			iris.append(_str)

	return iris

iris = importIrises()

class Attribute:
	def __init__(self, name, nodes):
		self.name = name
		self.nodesObject = nodes
		self.nodesKeys = list(nodes.keys())
		self.nodesKeys.sort()
		self.weights = []
		self.x_values = {}

		self.countWeights()

	def countWeights(self):
		first = self.nodesKeys[0]
		for nk in self.nodesKeys[1:]:
			weight = 1 - abs(float(first) - float(nk))/(float(self.nodesKeys[-1]) - float(self.nodesKeys[0]))
			self.weights.append({'from': first, 'to': nk, 'weight': weight})
			first = nk

	def countX(self, _init):
		init = str(_init)
		self.x_values[init] = 1
		ind = self.nodesKeys.index(init)
		arr_przed = list(reversed(self.nodesKeys[:ind]))
		arr_po = self.nodesKeys[ind+1:]
		for a in arr_przed:
			for w in self.weights:
				if w['from'] == a:
					_to = w['to']
					_weight = w['weight']
					self.x_values[a] = _weight * self.x_values[_to]
		for a in arr_po:
			for w in self.weights:
				if w['to'] == a:
					_from = w['from']
					_weight = w['weight']
					self.x_values[a] = _weight * self.x_values[_from]

	def __str__(self):
		return self.name + ': ' + str(self.x_values)

class Value:
	def __init__(self, value, indexes):
		self.value = value
		self.indexes = indexes
		self.x = 0.0

	def __str__(self):
		return '[' + str(self.indexes) + ']: ' + str(self.value)

	def __eq__(self, other):
		return self.value == other.value

def createValues(values, i, item):
	item_str = str(item)
	if not item_str in values:
		values[item_str] = []
	values[item_str].append(i)

my_iris = iris[2]


sie_values = {}
swi_values = {}
pie_values = {}
pwi_values = {}
class_values = {}

for (i, item) in enumerate(iris):
	createValues(sie_values, i, item[0])
	createValues(swi_values, i, item[1])
	createValues(pie_values, i, item[2])
	createValues(pwi_values, i, item[3])
	createValues(class_values, i, item[4])

param = [Attribute('sie', sie_values), Attribute('swi', swi_values), Attribute('pie', pie_values),
		 Attribute('pwi', pwi_values), Attribute('class', class_values)]

param[0].countX(my_iris[0])
param[1].countX(my_iris[1])
param[2].countX(my_iris[2])
param[3].countX(my_iris[3])
param[4].countX(my_iris[4])

iris_x = []

for ir in iris:
	x = param[0].x_values[str(ir[0])] * 0.2 + param[1].x_values[str(ir[1])] * 0.2 + param[2].x_values[str(ir[2])] * 0.2 +\
		param[3].x_values[str(ir[3])] * 0.2 + param[4].x_values[ir[4]] * 0.2
	iris_x.append(x)


print iris_x
