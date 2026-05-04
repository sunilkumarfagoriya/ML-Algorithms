import numpy as np
import pandas as pd

def gini(y):
    f = pd.DataFrame(y)
    total = f.value_counts()
    size = len(y)
    count = 0

    for i in total:
        count += ((i/size) ** 2)

    return 1 - count

def weighted_gini(left, right):
    size = len(left)+len(right)
    return len(left)/size * gini(left) + len(right)/size * gini(right)

def best_split(X, y):
    parent_gini = gini(y)
    best = 0
    name = None

    if(1 == len(set(y))):
        return None

    for col in X.columns:
        find_gini = dict()
        seen = list()

        for value in X[col].unique():
            find_gini[value] = list()

        for i in X[col].index:
            find_gini[X[col][i]].append(y[i])
        
        for key, value in find_gini.items():
            seen.append(value)

        score = parent_gini - weighted_gini(seen[0], seen[1])

        if(best < score):
            best = score
            name = col

    return name

def split(X, y, cols):
    seen = list()

    for feature in X[cols].unique():
        seen.append(feature)

    x_left, x_right, y_left, y_right = dict(), dict(), list(), list()

    for col in X.columns:
        x_left[col] = list()
        x_right[col] = list()

    for i in X[cols].index:
        if(seen[0] == X[cols][i]):
            for col in X.columns:
                x_left[col].append(X[col][i])
            y_left.append(y[i])
        else:
            for col in X.columns:
                x_right[col].append(X[col][i])
            y_right.append(y[i])
    
    x_left.pop(cols)
    x_right.pop(cols)

    return x_left, y_left, x_right, y_right

def build_tree(X, y):
    col = best_split(pd.DataFrame(X), y)
    
    if col is None:
        return max(set(list(y)), key=y.count)
    
    x_left, y_left, x_right, y_right = split(pd.DataFrame(X), y, col)

    left = build_tree(pd.DataFrame(x_left), y_left)
    right = build_tree(pd.DataFrame(x_right), y_right)

    return {
        'col': col,
        'left_val': X[col][0],
        'left': left,
        'right': right
    }

def predict_row(tree, row):
    if(type(tree) == str):
        return tree

    if(row[tree['col']] == tree['left_val']):
        return predict_row(tree['left'], row)
    else:
        return predict_row(tree['right'], row)


def predict(tree, data):
    result = []
    
    for i in data.index:
        row = data.iloc[i]
        result.append(predict_row(tree, row))
    
    return result








def bootstrap(X, y):
    size = len(y)

    indexes = np.random.randint(0, size, size)

    return indexes

def random_features(X, y):
    index = bootstrap(X, y)
    size = len(X.columns)

    features = np.random.choice(X.columns, size = int(np.sqrt(size)), replace = False)

    X_sample = X[features].iloc[index].reset_index(drop = True)
    y_sample = y.iloc[index].reset_index(drop = True)

    return X_sample, y_sample

def build_forest(bag):
    X = bag[0]
    y = bag[1]

    return build_tree(X, y)

data = pd.DataFrame({
    'Gender': ['Female','Female','Female','Female','Male','Male','Male','Male'],
    'Class':  [1, 1, 2, 2, 2, 2, 2, 2],
    'Survived': ['S','S','S','D','S','D','D','D']
})

X = data.drop('Survived', axis=1)
y = data['Survived']

model = list()

for i in range(5):
    bag = random_features(X, y)

    model.append(build_forest(bag))


result = list()

for tree in model:
    result.append(predict(tree, X))


final_predict = list()

for column in range(len(result[0])):
    votes = list()
    
    for row in range(len(result)):
        votes.append(result[row][column])

    final_predict.append(max(set(votes), key = votes.count))

print(final_predict)