def value_iteration():
	u = [0]*81
	uprime = [0]*81
	delta = -9999999
	discount = 0.99
	westMatrix = init_matrix("prob_west.txt")
	southMatrix = init_matrix("prob_south.txt")
	northMatrix = init_matrix("prob_north.txt")
	eastMatrix = init_matrix("prob_east.txt")
	f = open("rewards.txt", "r")
	rewards= []
	for line in f:
		line = float(line.rstrip('\n'))
		rewards.append(line)
	while(delta != 0):
		u = list(uprime)
		delta = 0
		for s in xrange(81):
			east = calculatesum(eastMatrix, s, u)
			west = calculatesum(westMatrix, s, u)
			north = calculatesum(northMatrix, s, u)
			south = calculatesum(southMatrix, s, u)
			uprime[s] = rewards[s] + discount * max(east, west, south, north)
			delta = max(delta, abs(uprime[s] - u[s]))
	print([value for value in u if value != 0])
	for s in xrange(81):
		d = {}
		if(u[s] > 0):
			if(s<72):
				d["East"] = calculatesum(eastMatrix, s, u)
			if(s>8):
				d["West"] = calculatesum(westMatrix, s, u)
			if(s%9 != 0):
				d["North"] = calculatesum(northMatrix, s, u)
			if(s%9 != 8):
				d["South"] = calculatesum(southMatrix, s, u)
			print("(" + str(s+1) + ", " + str(u[s]) + ", " + max(d, key=d.get) + ")")
	return u

def init_matrix(file):
	matrix = [[0.0 for x in xrange(81)] for y in xrange(81)]
	f = open(file, "r")
	for line in f.readlines():
		s, s_, prob = map(float,line.strip().split())
		matrix[int(s) - 1][int(s_) - 1] = prob
	return matrix

def calculatesum(matrix, s, u):
	num = 0
	for y in xrange(81):
		if (matrix[s][y] != 0.0):
			num += matrix[s][y]*u[y]
	return num

value_iteration()