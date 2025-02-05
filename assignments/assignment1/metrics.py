def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
    assert len(prediction) == len(ground_truth), "predictions should be the same length as gt"
    
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i] == 1:
            tp += 1
        elif prediction[i] == 1 and ground_truth[i] == 0:
            fp += 1
        elif prediction[i] == 0 and ground_truth[i] == 1:
            fn += 1
            
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    
    accuracy = (prediction == ground_truth).sum() / len(prediction)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = (prediction == ground_truth).sum() / len(prediction)
    
    return accuracy
