import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for layer, weights in self.params().items():
          weights.grad = np.zeros_like(weights.grad)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        fc1_out = self.fc1.forward(X)
        relu_out = self.relu.forward(fc1_out)
        preds = self.fc2.forward(relu_out)

        loss, dpred = softmax_with_cross_entropy(preds, y)
        drelu_out = self.fc2.backward(dpred)
        dfc1_out = self.relu.backward(drelu_out)
        dinput = self.fc1.backward(dfc1_out)
        
        sum_reg_loss = 0
        for layer, weights in self.params().items():
          reg_loss, reg_grad = l2_regularization(weights.value, self.reg)
          weights.grad += reg_grad
          sum_reg_loss += reg_loss
        
        loss += sum_reg_loss
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        fc1_out = self.fc1.forward(X)
        relu_out = self.relu.forward(fc1_out)
        pred_raw = self.fc2.forward(relu_out)
        pred = np.argmax(pred_raw, axis=1)
        
        return pred

    def params(self):
      # TODO Implement aggregating all of the params
      result = {'fc1': self.fc1.params()['W'], 'fc2': self.fc2.params()['W']}
      return result
