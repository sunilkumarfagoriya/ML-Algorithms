import numpy as np

def gaussian_score(X, mean, variance):
    gau_score = list()
    
    for i in range(len(X)):
        gau_score.append(np.exp(-((X[i] - mean) ** 2) / (2 * variance)))
    
    return np.array(gau_score)


def initialize_params(X, k):
    k_mean = np.random.choice(X, k, replace = False)
    k_variance = np.full(k, np.var(X))
    k_weight = np.ones(k) / k

    return k_mean, k_variance, k_weight


def e_step(X, k_mean, k_variance, k_weight, k):
    k_gau_score = list()

    for i in range(k):
        k_gau_score.append(gaussian_score(X, k_mean[i], k_variance[i]))
    
    for i in range(k):
        k_gau_score[i] *= k_weight[i]

    for i in range(len(X)):
        total_proba = 0

        for j in range(k):
            total_proba += k_gau_score[j][i]
        
        for j in range(k):
            k_gau_score[j][i] /= total_proba
        
    return k_gau_score


def m_step(X, proba_metrix, k):
    k_mean = list()

    for i in range(k):
        if(len(X.shape) > 1):
            mean = np.zeros(X.shape[1])
        else:
            mean = 0

        for j in range(len(X)):
            mean += (X[j] * proba_metrix[i][j])
        
        mean /= np.sum(proba_metrix[i])

        k_mean.append(mean)

    k_variance = list()

    for i in range(k):
        if(len(X.shape) > 1):
            variance = np.zeros(X.shape[1])
        else:
            variance = 0

        for j in range(len(X)):
            variance += (proba_metrix[i][j] * ((X[j] - k_mean[i])**2))
        
        variance /= np.sum(proba_metrix[i])

        k_variance.append(variance)

    k_weight = list()

    for i in range(k):
        weight = np.sum(proba_metrix[i])/len(X)
        k_weight.append(weight)

    return k_mean, k_variance, k_weight


def gmm(X, k, max_iter):
    k_mean, k_variance, k_weight = initialize_params(X, k)

    for i in range(max_iter):
        proba_matrix = e_step(X, k_mean, k_variance, k_weight, k)

        new_mean, new_variance, new_weight = m_step(X, proba_matrix, k)

        if(np.allclose(k_mean, new_mean)):
            break

        k_mean, k_variance, k_weight = new_mean, new_variance, new_weight

    return k_mean, k_variance, k_weight, proba_matrix


np.random.seed(42)

group1 = np.random.randn(50) * 1.0 + 2.0
group2 = np.random.randn(50) * 1.0 + 8.0

X = np.concatenate([group1, group2])

k = 2

max_iter = 100

print(gmm(X, k, max_iter))
