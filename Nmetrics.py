from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score, accuracy_score
from sklearn.metrics import pairwise_distances
import math
import numpy as np


nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score


def acc(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    return accuracy_score(y_true, y_voted_labels)


def cluster_metrics(X, k, labels, centers):
    intra_cluster_distances = []
    for i in range(k):
        cluster_samples = X[labels == i]
        if len(cluster_samples) <= 1:
            intra_cluster_distances.append(0)
        else:
            intra_cluster_distances.append(np.mean(pairwise_distances(cluster_samples)))


    inter_cluster_distances = pairwise_distances(centers)
    inter_cluster_distance_disparity = np.std(inter_cluster_distances)


    compactness = np.mean(intra_cluster_distances)


    separation = np.mean(inter_cluster_distances)

    cluster_densities = [len(X[labels == i]) / len(X) for i in range(k)]
    density = np.mean(cluster_densities)

    return inter_cluster_distance_disparity, compactness, separation, density


def calculate_entropy(counter):
    total = sum(counter.values())
    entropy = 0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return entropy

def calculate_variance(counter):
    counts = list(counter.values())
    return np.var(counts)

def calculate_gini(counter):
    total = sum(counter.values())
    counts = np.array(list(counter.values()))
    counts_sorted = np.sort(counts)
    n = len(counts)
    cumulative_counts = np.cumsum(counts_sorted)
    gini = (n + 1 - 2 * np.sum(cumulative_counts) / total) / n
    return gini