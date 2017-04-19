from __future__ import print_function
import math
import csv
import random

random_range = [-0.1, 0.1]

with open('iris.csv', 'rb') as csvfile:
	iris_dataset = csv.reader(csvfile, delimiter=',')
	iris = []
	for row in iris_dataset:
		_str = []
		for i in row:
			try:
				float(i)
				_str.append(float(i))
			except ValueError:
				if i == 'Iris-setosa':
					_str.append(1.0)
					_str.append(0.0)
					_str.append(0.0)
				elif i == 'Iris-versicolor':
					_str.append(0.0)
					_str.append(1.0)
					_str.append(0.0)
				elif i == 'Iris-virginica':
					_str.append(0.0)
					_str.append(0.0)
					_str.append(1.0)
		iris.append(_str)

	random.shuffle(iris)


class Input:
	def __init__(self, _id, _sum):
		self._id = _id
		self.y = _sum
		self.output_connections = []

	# def __str__(self):
	# 	return str(self._id)

	# def print_neuron(self):
	# 	print("Input  _id={}:\t\t|\t".format(self._id), end='')
	# 	for o in self.output_connections:
	# 		print("{} ".format(o), end='')
	# 	print("")


class Neuron:
	def __init__(self, _id, input_connections, desired_output=None):
		self._id = _id
		self.input_connections = input_connections
		self.output_connections = []
		self.input_weights = []
		self.output_weights = []
		self._sum = 0
		self.delta = 0
		self.y = 0
		self.desired_output = desired_output
		self.bias = 1.0
		self.learningWeight = 0.1
		self.biasWeight = random.uniform(random_range[0], random_range[1])

		self.compute_random_weights()
		self.set_this_neuron_in_antecedent_output()
		# self.compute_sum()
		# self.compute_my_function()

	# def __str__(self):
	# 	return str(self._id)

	def compute_random_weights(self):
		for r in self.input_connections:
			self.input_weights.append(random.uniform(random_range[0], random_range[1]))

	def set_this_neuron_in_antecedent_output(self):
		for (i, obj) in enumerate(self.input_connections):
			if isinstance(obj, Neuron):
				obj.output_connections.append(self)
				obj.output_weights.append(self.input_weights[i])

	# def print_neuron(self):
	# 	print("Neuron _id={}:\t".format(self._id), end='')
	# 	for i in self.input_connections:
	# 		print("{} ".format(i), end='')
	# 	print("  |\t", end='')
	# 	for o in self.output_connections:
	# 		print("{} ".format(o), end='')
	# 	print(" // _sum={} // Y={} //Delta={}".format(str(self._sum), str(self.y), str(self.delta)))

	def compute_sum(self):
		self._sum = 0
		for i in range(0, len(self.input_connections)):
			#if isinstance(self.input_connections[i], Neuron):
			self._sum += self.input_connections[i].y * self.input_weights[i]
		self._sum += self.biasWeight

	def compute_my_function(self):
		self.y = 1 / (1 + math.exp(-1 * self._sum))

	def derivative(self):
		return (1.0 - self.y) * self.y

	def compute_delta(self):
		if self.desired_output is not None:
			self.delta = (self.desired_output - self.y)
		else:
			self.delta = 0
			for i in range(0, len(self.output_connections)):
				self.delta += self.output_connections[i].delta * self.output_weights[i] * self.output_connections[i].derivative()

	def improve_weights(self):
		self.biasWeight += self.learningWeight * self.delta * self.bias * self.derivative()
		for i in range(0, len(self.input_weights)):
			if isinstance(self.input_connections[i], Neuron):
				self.input_weights[i] += self.learningWeight * self.delta * self.input_connections[i].y

count_ok = 0
iterations = 500
division_point = 75

# training -------------------------------------------------------------------------------------------------------------

inputs = [Input(1, 0.0), Input(2, 0.0), Input(3, 0.0), Input(3, 0.0)]
layer1 = [Neuron(5, inputs), Neuron(6, inputs), Neuron(7, inputs), Neuron(8, inputs),Neuron(5, inputs), Neuron(6, inputs), Neuron(7, inputs), Neuron(8, inputs)]
#layer2 = [Neuron(8, layer1), Neuron(4, layer1), Neuron(9, layer1), Neuron(10, layer1)]
layer3 = [Neuron(11, layer1, 0.0), Neuron(12, layer1, 0.0), Neuron(13, layer1, 0.0)]

network = [inputs, layer1, layer3]

for s in range(iterations):
	for item in iris[0:division_point]:
		inputs = item[0:4]
		outputs = item[4:7]

		for (i, obj) in enumerate(network[0]):
			obj.y = inputs[i]

		for (i, obj) in enumerate(network[len(network)-1]):
			obj.desired_output = outputs[i]

		for n in network:
			for l in n:
				if isinstance(l, Neuron):
					l.compute_sum()
					l.compute_my_function()

		for n in reversed(network):
			for l in reversed(n):
				if isinstance(l, Neuron):
					l.compute_delta()

		for n in reversed(network):
			for l in reversed(n):
				if isinstance(l, Neuron):
					l.improve_weights()


# testing --------------------------------------------------------------------------------------------------------------

for item in iris[division_point:len(iris)-1]:
	inputs = item[0:4]
	outputs = item[4:7]

	for (i, obj) in enumerate(network[0]):
		obj.y = inputs[i]

	for n in network:
		for l in n:
			if isinstance(l, Neuron):
				#l.compute_random_weights()
				l.compute_sum()
				l.compute_my_function()

	print("Desired outputs: {}".format(outputs))

	for l in network:
		if isinstance(l[len(l) - 1], Neuron) and l == network[len(network) - 1]:
			print("output:", l[0].y, l[1].y, l[2].y)
			_max = l[0].y
			ind = 0
			if l[1].y > _max:
				ind = 1
			if l[2].y > _max:
				ind = 2

			max_des = outputs[0]
			ind_des = 0
			if outputs[1] > max_des:
				ind_des = 1
			if outputs[2] > max_des:
				ind_des = 2

			if ind == ind_des:
				count_ok += 1

result = float(count_ok)/len(iris[division_point:len(iris)-1])*100
print("ok = {:.2f} %".format(result))
