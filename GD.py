import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, Y, theta, alpha, numIterations):
	'''
	# This function returns a tuple (theta, Cost array)
	'''
	m = len(Y)
	Y = [ i[0] for i in Y]
	arrCost =[];
	transposedX = np.transpose(X)

	for interation in range(0, numIterations):
		if interation%100 == 0:
			print(interation/100)
		prediction = np.dot(X, theta)
		residualError = np.subtract(prediction, Y)
		#residualError = np.subtract(np.dot(X, theta), Y)
		gradient =  np.array([sum(x)/m for x in residualError * transposedX])
		
		change = [alpha * x for x in gradient]
		theta = np.subtract(theta, change)  # theta = theta - alpha * gradient
		
		atmp = np.sum(residualError ** 2) / m
		arrCost.append(atmp)

	return theta, arrCost
