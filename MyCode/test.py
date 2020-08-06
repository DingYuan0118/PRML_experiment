from BayesEstimate import data_generator
import numpy as np
mean_a = [-1, 0]
mean_b = [1, 0]
cova = [[1, 0.5], [0.5, 1]]
covb = [[1, -0.5], [-0.5, 1]]
num = 1000
classa, classb = data_generator(mean_a, mean_b, cova, covb, num, num)
data = {1: classa, 2: classb}
x = np.array(([1,2],[3,4]), dtype=float)
print(x**-1)