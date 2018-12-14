import matplotlib.pyplot as plt
import numpy as np

def readFile():
	res = []
	with open('accuracy.txt', 'r') as f:
		lines = f.readlines()
		for line in lines:
			data = line.split()[1]
			res.append(float(data))
	return res

x = np.arange(28)
data = readFile()
print(data)
plt.bar(x, data)
axes = plt.gca()
axes.set_xlim([-1,28])
axes.set_ylim([0,1])
fig = plt.gcf()
fig.canvas.set_window_title('Accuracy for all classifiers')
plt.xlabel('# of classifier')
plt.ylabel('Accuracy')
plt.show()
