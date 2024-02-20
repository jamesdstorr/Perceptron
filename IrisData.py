#Import the Iris Dataset 
import pandas as pd
s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
print("From URL:", s )
df = pd.read_csv(s, header=None, encoding='utf-8')
print(df.head())

 
import matplotlib.pyplot as plt
import numpy as np
# Extract the first 100 rows of data from the datasource, which contains Iris Setosa and Iris Veriscolor samples
y = df.iloc[0:100, 4].values
# Convert the Iris Setosa and Iris Veriscolor to 0 and 1, respectively.
y = np.where(y == 'Iris-setosa', 0, 1)
# Extract sepal length and petal length
# In the datset, column 0 is Sepal Length and column 2 is Petal Length 
X = df.iloc[0:100, [0, 2]].values
# Plot data - the first 50 values are Iris Setosa and the next 50 are Iris Veriscolor

plt.figure(num="Iris Data Scatter Plot",figsize=(8, 4))
plt.scatter(X[:50, 0], X[:50, 1],
                        color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
                        color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')




#Train the model
from Perceptron import Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.figure(num="Perceptron Errors over Epochs",figsize=(8, 4))
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')




# visualise the decision boundary 
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.figure(num="Decision Boundary", figsize=(8, 5))
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()