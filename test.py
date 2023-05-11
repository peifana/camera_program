import torch
from matplotlib import pyplot
A=torch.reshape(torch.range(0,5),(2,3))
B=torch.reshape(torch.range(0,5),(2,3))
pyplot.plot(A,B)
pyplot.show()