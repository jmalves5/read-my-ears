import evaluate
import numpy as np

def compute_metrics(eval_pred):
    """Compute accuracy metric for evaluation."""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
