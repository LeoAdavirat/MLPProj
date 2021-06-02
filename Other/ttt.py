	def BigProblem(n):
		Sum = 0
		for i in range(1, n):
			if i % 2 == 0:
				Sum += i
			else:
				Sum += i * 2
		return Sum