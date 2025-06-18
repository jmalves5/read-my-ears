import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
# Given a vector of binary labels and a vector of predicted binary labels, we calculate the following metrics: precision, recall, f1 score, accuracy, and confusion matrix.
# We also calculate the area under the ROC curve and the area under the precision-recall curve.
# We also calculate the average precision score.
#

def calculate_metrics(y_true, y_pred):
    # Calculate precision, recall, f1 score, accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return precision, recall, f1, accuracy, cm

def precision_score(y_true, y_pred):
    tp = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1])
    fp = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1])

    if tp == 0 and fp == 0:
        return 0

    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    tp = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1])
    fn = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0])

    if tp == 0 and fn == 0:
        return 0

    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision == 0 and recall == 0:
        return 0
   
    return 2 * (precision * recall) / (precision + recall)

def accuracy_score(y_true, y_pred):
    correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred):
    cm = [[0, 0],  # Actual 0: [True Negatives (TN), False Positives (FP)]
          [0, 0]]  # Actual 1: [False Negatives (FN), True Positives (TP)]
    for actual, predicted in zip(y_true, y_pred):
        if actual == 0 and predicted == 0:
            cm[0][0] += 1  # True Negative
        elif actual == 0 and predicted == 1:
            cm[0][1] += 1  # False Positive
        elif actual == 1 and predicted == 0:
            cm[1][0] += 1  # False Negative
        else:  # actual == 1 and predicted == 1
            cm[1][1] += 1  # True Positive

    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='coolwarm')
    fig.colorbar(cax)
    

    plt.xlabel('Predicted')
    plt.ylabel('True')
    wandb.log({
        "test_confusion_matrix_plot": fig,
    })
    plt.close("all")
    return cm

def roc_curve(y_true, y_pred):
    fpr = []
    tpr = []
    thresholds = []
    for threshold in np.linspace(0, 1, 100):
        thresholds.append(threshold)
        y_pred_binary = [1 if y > threshold else 0 for y in y_pred]
        fpr.append(1 - recall_score(y_true, y_pred_binary))
        tpr.append(recall_score(y_true, y_pred_binary))
    return fpr, tpr, thresholds