### Splitting mnist dataset

import pandas as pd

train = pd.read_csv('mnist_train.csv', header=None)
test = pd.read_csv('mnist_test.csv', header=None)


# Splitting features and labels in mnist_train
# x_train = features
# y_train = labels
x_tr = train.iloc[:,1:]
y_tr = train.iloc[:,:1]

# Splitting features and labels in mnist_test
# x_test = features
# y_test = labels
x_ts = test.iloc[:,1:]
y_ts = test.iloc[:,:1]

# Saving to separate CSVs for ease of later use.
x_tr.to_csv('x_train.csv',index=False, header=False)
y_tr.to_csv('y_train.csv', index=False, header=False)

x_ts.to_csv('x_test.csv', index=False, header=False)
y_ts.to_csv('y_test.csv', index=False, header=False)

print('done')
