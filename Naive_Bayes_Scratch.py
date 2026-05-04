import numpy as np
import pandas as pd

def class_prob(y):
    prob = y.value_counts().astype(float)
    for key, value in prob.items():
        prob[key] /= len(y)

    return prob

def feature_prob(X, y):
    features_prob = dict()
    
    for key, value in X.items():
        prob = dict()
        features_prob[key] = prob
        for i in np.unique(X[key]):
            prob[i] = dict()
            for j in np.unique(y):
                prob[i][j] = 0

        for i in range(len(X[key])):
            if(y[i] in prob[X[key][i]]):
                prob[X[key][i]][y[i]] += 1

        for vals in prob:
            class_count = y.value_counts()
            for val in prob[vals]:
                prob[vals][val] = float((prob[vals][val] + 1) / (class_count[val] + len(np.unique(X[key]))))

    return features_prob


def predict(X_test, class_prob, feature_prob):
    y_pred = list()
    
    for i in range(len(X_test)):
        prob = list()
        cla_name = "a"
        max_prob = float(0)

        for cls in class_prob.index:
            total = float(class_prob[cls])
            
            for col in X_test.columns:
                total *= feature_prob[col][X_test[col][i]][cls]
            
            if(max_prob < total):
                cla_name = cls
                max_prob = total
        
        y_pred.append(cla_name)

    return y_pred



data = pd.DataFrame({
    'Weather': ['Sunny','Sunny','Cloudy','Rainy','Rainy','Cloudy'],
    'Play':    ['No','No','Yes','Yes','Yes','Yes']
})

X = data[['Weather']]
y = data['Play']

target_prob = class_prob(y)

features_prob = feature_prob(X, y)

X_test = pd.DataFrame({
    'Weather': ['Sunny', 'Rainy', 'Cloudy']
})

print(predict(X_test, target_prob, features_prob))

