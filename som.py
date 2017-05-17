import random
import math
from neuralNetwork import Neuron, Input, importIrises
from PIL import Image, ImageDraw

random_range = [0, 1]
iris = importIrises()
A = 1000

class Node(Neuron):
	def __init__(self, i, j, _inputs):
		Neuron.__init__(self, _inputs)
		self.i = i
		self.j = j
		self.weights = [] # 4x4 jak inputy
		self._inputs = _inputs
		self.min_input = min(map(lambda inp: inp.y, _inputs))
		self.max_input = max(map(lambda inp: inp.y, _inputs))
		self.colors = [0, 0, 0]
		self.sum = 0
		self.radius = 4
		self.learningRate = 1
		self.initialize_weights()

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
		self.weights = map(lambda (syn, weight): weight + delta * learning_rate * (syn.y - weight),
								zip(self._inputs, self.weights))

	def __str__(self):
		return ", ".join(map(lambda color: str(color), self.colors))


class Grid:
	def __init__(self, _inputs, size):
		self.size = size
		self._inputs = _inputs
		self.grid = []
		self.time = 0
		self.create_grid()

	def create_grid(self):
		for index_i in range(self.size):
			self.grid.append([])
			for index_j in range(self.size):
				self.grid[index_i].append(Node(index_i, index_j, self._inputs))

	def find_winner(self):
		distances = []
		for row in self.grid:
			for node in row:
				_sum = 0
				for (i, inp) in enumerate(self._inputs):
					_sum += math.pow(inp.y - node.weights[i], 2)
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

print grid

grid.draw_grid()









