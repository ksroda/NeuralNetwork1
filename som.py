import random
import math
import csv
# from neuralNetwork import Neuron, Input, importIrises
from PIL import Image, ImageDraw

random_range = [0, 1]
A = 1000

# -----------------------------------------------------------------------------
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
	return iris

iris = importIrises()

class Input:
	def __init__(self, _sum):
		self.y = _sum
		self.output_connections = []

class Neuron:
	def __init__(self, input_connections, desired_output=None):
		self.input_connections = input_connections
		self.output_connections = []
		self.input_weights = []
		self.output_weights = []
		self._sum = 0
		self.delta = 0
		self.y = 0
		self.desired_output = desired_output
		self.bias = 1.0
		self.learningWeight = 0.5
		self.biasWeight = random.uniform(random_range[0], random_range[1])

		self.compute_random_weights()
		self.set_this_neuron_in_antecedent_output()

	def compute_random_weights(self):
		for r in self.input_connections:
			self.input_weights.append(random.uniform(random_range[0], random_range[1]))

	def set_this_neuron_in_antecedent_output(self):
		for (i, obj) in enumerate(self.input_connections):
			if isinstance(obj, Neuron):
				obj.output_connections.append(self)
				obj.output_weights.append(self.input_weights[i])

	def compute_sum(self):
		self._sum = 0
		for i in range(0, len(self.input_connections)):
			#if isinstance(self.input_connections[i], Neuron):
			self._sum += self.input_connections[i].y * self.input_weights[i]
		self._sum += self.biasWeight
		#self._sum += self.bias

	def compute_my_function(self):
		self.y = 1 / (1 + math.exp(-1 * self._sum))
		#self.y = math.tanh(self._sum)

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

# -----------------------------------------------------------
class Node(Neuron):
	def __init__(self, i, j, _inputs):
		Neuron.__init__(self, _inputs)
		self.i = i
		self.j = j
		# self.weights = [] # 4x4 jak inputy
		self._inputs = _inputs
		self.min_input = min(map(lambda inp: inp.y, _inputs))
		self.max_input = max(map(lambda inp: inp.y, _inputs))
		self.colors = [0, 0, 0]
		self.sum = 0
		self.radius = 4
		self.learningRate = 1
		# self.initialize_weights()

	def initialize_weights(self):
		for index in range(len(self._inputs)):
			_from = (self.max_input - self.min_input)/100
			_to = (self.max_input - self.min_input)/10
			self.weights.append(random.uniform(_from, _to))

	def improve_node_weights(self, distance, time):
		radius = self.radius * math.exp(-time / A)
		learning_rate = self.learningRate * math.exp(-time / A)
		k = distance ** 2 / (2 * (radius ** 2))
		delta = math.exp(-k)
		# self.weights = map(lambda (syn, weight): weight + delta * learning_rate * (syn.y - weight),
		# 						zip(self._inputs, self.weights))
		self.input_weights = map(lambda (syn, weight): weight + delta * learning_rate * (syn.y - weight),
								zip(self._inputs, self.input_weights))
	def __str__(self):
		return ", ".join(map(lambda color: str(color), self.colors))


class Grid:
	def __init__(self, _inputs, size):
		self.size = size
		self._inputs = _inputs
		self.grid = []
		self.time = 0
		self.layer = []
		self.create_grid()

	def create_grid(self):
		for index_i in range(self.size):
			self.grid.append([])
			for index_j in range(self.size):
				node = Node(index_i, index_j, self._inputs)
				self.grid[index_i].append(node)
				self.layer.append(node)

	def find_winner(self):
		distances = []
		for row in self.grid:
			for node in row:
				_sum = 0
				for (i, inp) in enumerate(self._inputs):
					_sum += math.pow(inp.y - node.input_weights[i], 2)
				_sum = math.sqrt(_sum)
				distances.append({"sum": _sum, "node": node})
		return min(distances, key=lambda d: d["sum"])["node"]

	def learn(self, input, outputs):
		for (i, obj) in enumerate(self._inputs):
			obj.y = input[i]
		winner = self.find_winner()
		which_iris = outputs.index(max(outputs))
		winner.colors[which_iris] += 1
		for row in self.grid:
			for node in row:
				distance = abs(node.i - winner.i) + abs(node.j - winner.j)
				node.improve_node_weights(distance, self.time)
		self.time += 1

	def __str__(self):
		result = ""
		for row in self.grid:
			result += " | "
			for node in row:
				result += str(node)
				result += " | "
			result += "\n"
		return result

	def draw_grid(self):
		im = Image.new('RGBA', (600, 600), (0, 0, 0, 0))
		draw = ImageDraw.Draw(im)
		for (i, row) in enumerate(self.grid):
			for (j, node) in enumerate(row):
				_sum = sum(node.colors)
				r = int((node.colors[0] / float(_sum)) * 255)
				g = int((node.colors[1] / float(_sum)) * 255)
				b = int((node.colors[2] / float(_sum)) * 255)
				draw.rectangle([60*i, 60*j, 60*i+50, 60*j+50], fill=(r, g, b))
		im.show()

	def to_mlp(self, inp):
		for node in self.layer:
			suma = 0.0
			for _inp,_w in zip(inp,node.input_weights):
				suma = suma + math.pow((_w - _inp),2)
			node.y = math.sqrt(suma)

inputs = [Input(1), Input(2), Input(0.5), Input(1.5)]
grid = Grid(inputs, 5)
iris_data = []

for i in range(100):
	iris_data += iris

random.shuffle(iris_data)

for item in iris_data:
	input = item[0:4]
	output = item[4:7]
	grid.learn(input, output)

# print grid
# grid.draw_grid()

count_ok = 0
iterations = 300
division_point = 60

# training -------------------------------------------------------------------------------------------------------------

# inputs = [Input(0.0), Input(0.0), Input(0.0), Input(0.0)]
layer1 = [Neuron(inputs), Neuron(inputs), Neuron(inputs), Neuron(inputs), Neuron(inputs), Neuron(inputs),
				Neuron(inputs), Neuron(inputs), Neuron(inputs), Neuron(inputs)]
#layer2 = [Neuron(8, layer1), Neuron(4, layer1), Neuron(9, layer1), Neuron(10, layer1)]
layer3 = [Neuron(layer1, 0.0), Neuron(layer1, 0.0), Neuron(layer1, 0.0)]

network = [grid.layer, layer1, layer3]

for s in range(iterations):
	for item in iris[0:division_point]:
		inputs = item[0:4]
		outputs = item[4:7]
		grid.to_mlp(inputs)

		# for (i, obj) in enumerate(network[0]):
		# 	obj.y = inputs[i]

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
	grid.to_mlp(inputs)

	# for (i, obj) in enumerate(network[0]):
	# 	obj.y = inputs[i]

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
