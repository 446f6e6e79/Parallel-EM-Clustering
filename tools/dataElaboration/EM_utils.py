import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def cluster_mapping(y_true, y_pred):
    """
    Return:
      - pred_to_real: dict mapping predicted_label -> real_label
      - real_to_pred: dict mapping real_label -> predicted_label
      - accuracy: permutation-invariant accuracy in [0,1]
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    # Use same label order on rows/cols (union), so we can map indices back to labels
    labels = np.unique(np.concatenate([labels_true, labels_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_ind, col_ind = linear_sum_assignment(-cm)

    pred_to_real = {labels[c]: labels[r] for r, c in zip(row_ind, col_ind) if labels[c] in set(labels_pred)}
    real_to_pred = {labels[r]: labels[c] for r, c in zip(row_ind, col_ind) if labels[r] in set(labels_true)}
    acc = (cm[row_ind, col_ind].sum() / cm.sum()) if cm.sum() > 0 else 0.0
    return pred_to_real, real_to_pred, acc

def clustering_accuracy(df):
    """
        Calculate clustering accuracy from a DataFrame containing 'predicted' and 'real' columns.
        Uses the Hungarian algorithm to find the best matching between predicted and real labels.
        Parameters:
            df: DataFrame containing the clustering results
        Returns:
            accuracy: Clustering accuracy as a float
    """
    y_pred = df['predicted_cluster'].to_numpy()
    y_true = df['real_cluster'].to_numpy()
    _, _, accuracy = cluster_mapping(y_true, y_pred)
    return accuracy

def derive_cluster_mapping(df):
    """
    Convenience wrapper using DataFrame columns 'predicted_cluster' and 'real_cluster'.
    """
    y_pred = df['predicted_cluster'].to_numpy()
    y_true = df['real_cluster'].to_numpy()
    return cluster_mapping(y_true, y_pred)

def remap_predicted(y_pred, pred_to_real):
    """
    Remap predicted labels into the real label space using the mapping.
    """
    return np.array([pred_to_real.get(p, p) for p in y_pred])