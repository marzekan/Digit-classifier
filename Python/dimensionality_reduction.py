import pandas as pd
import csv
from sklearn.decomposition import PCA

train_csv = pd.read_csv("train_nolab.csv", delimiter=',')
pca = PCA(n_components = 0.95)
reduced_train_csv = pca.fit_transform(train_csv)

with open("mnist_train_154.csv", "w") as file:
    writer = csv.writer(file, delimiter=',')
    
    for line in reduced_train_csv:
        writer.writerow(line)
