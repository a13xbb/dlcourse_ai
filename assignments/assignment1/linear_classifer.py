import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    preds = predictions.copy()
    preds -= np.max(preds, axis=-1)[..., None]
    denominator = np.sum(np.exp(preds), axis=-1)[..., None]
    sm = np.exp(preds) / denominator
    # Your final implementation shouldn't have any loops
    return sm


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    loss = 0
    if type(target_index) is int:
      loss = -np.log(probs[target_index])
    else:
      for i, sample in enumerate(probs):
        loss -= np.log(sample[target_index[i]])
    # Your final implementation shouldn't have any loops
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    dprediction = probs.copy()
    
    if type(target_index) is int:
      dprediction[target_index] = probs[target_index] - 1
    else:
      for i in range(len(predictions)):
        dprediction[i][target_index[i]] = probs[i][target_index[i]] - 1

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W 

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    dW = np.zeros_like(W)
    # TODO implement prediction and gradient over W
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            batches = []
            for batch_indices in batches_indices:
              batches.append((X[batch_indices], y[batch_indices]))
            
            loss = 0
            for X_batch, y_batch in batches:
              ce_loss, dw_grad = linear_softmax(X_batch, self.W, y_batch)
              l2_loss, l2_grad = l2_regularization(self.W, reg)
              loss += ce_loss + l2_loss
              self.W -= learning_rate * (dw_grad + l2_grad)
              
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            loss_history.append(loss / batch_size)
            # end
            if (epoch + 1) % 10 == 0:
              print("Epoch %i, loss: %f" % (epoch + 1, loss / batch_size))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=int)

        # TODO Implement class prediction
        y_preds_raw = np.dot(X, self.W)
        y_pred = np.argmax(y_preds_raw, axis=1)
        # Your final implementation shouldn't have any loops
        return y_pred



                
                                                          

            

                
