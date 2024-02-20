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
plt.scatter(X[:50, 0], X[:50, 1],
                        color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
                        color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


#Train the model
from Perceptron import Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
