# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

""""
getting help from ChatGPT and Isaiah Hazelwood (Biophysics student)
"""

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            #creates weights from a standard normal distribution (mean = 0, std = 1) and scales them down by multiplying by 0.1.
            #generates a matrix of shape output_dim, input_dim, because need to connect every input neuron to an output neuron. 
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = A_prev @ W_curr.T + b_curr.T
        if activation == "relu":
            A_curr = self._relu(Z_curr)
        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        else:
            raise ValueError(f"Unknown activation {activation}")
        return A_curr, Z_curr


    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.

        _single_forward() is being called for each layer. can think of it as the method looping through all layers in self.arch. 
        """
        curr_A = X
        """
        in the cache I store intermediate values (like Z and A for each layer) during the forward pass. 
        Because during backpropagation, I need these intermediate values to compute the gradients. I need Z (the linear outputs) to compute derivatives 
        of activation functions. I need A_prev (activations from previous layer) to compute gradients for weights: dW = dZ @ A_prev.T
        """

        cache = {}
        #store input as A0
        cache["A0"] = curr_A
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            next_A, next_Z = self._single_forward(self._param_dict[f"W{layer_idx}"],
                                                  self._param_dict[f"b{layer_idx}"],
                                                  curr_A, 
                                                  layer["activation"])
            cache[f"Z{layer_idx}"] = next_Z
            cache[f"A{layer_idx}"] = next_A
            curr_A = next_A
        return curr_A, cache
   

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        #dZ is the dLoss/dZ = dL/dA * activation of Z, so the contribution of the activation function to how the loss changes
        if activation_curr == "relu":
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

        elif activation_curr == "sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise ValueError(f"Unknown activation: {activation_curr}")
        
        m = A_prev.shape[0] # batch size
        

        #tells me how much the loss would change if I change each weight slightly. 
        #for every weight between neuron i and neuron j it says how much would it change the final loss if I tweak the weight up or down a bit. 
        #this it the key ingredient for updating the weights. W = W - learning_rate * dW
        dW_curr = dZ_curr.T @ A_prev / m

        #db is the sum of dZ across batch (here only one layer, no batch)
        #how much did shifting the bias up or down affect the loss? b = b - learning_rate * db
        db_curr = np.mean(dZ_curr, axis=0, keepdims=True).T

        #tells me how much the output A_prev influenced the loss. this is the signal I send back to the previous layer. 
        #this is the starting dA for the previous layer in my next single_backprop call. 
        dA_prev = dZ_curr @ W_curr

        return dA_prev, dW_curr, db_curr


    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        if self._loss_func == "mean_square_error":
            curr_dA = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func == "binary_cross_entropy":
            curr_dA = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            raise ValueError(f"Unknown loss function {self._loss_func}")
        
        grad_dict = {}

        #loop backwards through layers
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1

            W_curr = self._param_dict[f"W{layer_idx}"]
            b_curr = self._param_dict[f"b{layer_idx}"]
            Z_curr = cache[f"Z{layer_idx}"]
            A_prev = cache[f"A{idx}"] #A0 is input, A1 is after layer 1... 
            activation = layer["activation"]

            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr, 
                b_curr, 
                Z_curr, 
                A_prev, 
                curr_dA, 
                activation
            )

            #save gradients
            grad_dict[f"W{layer_idx}"] = dW_curr
            grad_dict[f"b{layer_idx}"] = db_curr

            #move one layer back 
            curr_dA = dA_prev
        
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """

        #we want to update the weights based on the average error across all examples in the batch, not just one of them. 

        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1
            #to minimize loss, we go in the opposite direction. the gradient tells us the direction of the steepest increase.
            #so we update weights like: W = W - learning_rate * dW. same for biases. the learning rate defines how big of a step we should go in the opposite direction.
            self._param_dict[f"W{layer_idx}"] -= self._lr * np.mean(grad_dict[f"W{layer_idx}"])
            self._param_dict[f"b{layer_idx}"] -= self._lr * np.mean(grad_dict[f"b{layer_idx}"])


    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        #epoch is one complete pass through the entire training dataset
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        for epoch_i in range(self._epochs):
            for batch_i in range(X_train.shape[0] // self._batch_size):
                batch_start = batch_i * self._batch_size
                batch_stop = (batch_i + 1) * self._batch_size 
                X_train_batch = X_train[batch_start:batch_stop,:]
                y_train_batch = y_train[batch_start:batch_stop, :]
                y_pred_batch, forward_cache = self.forward(X_train_batch)
                batch_grads = self.backprop(y_train_batch, y_pred_batch, forward_cache)
                self._update_params(batch_grads)

            #calculating the accuracies for this epok
            y_train_pred = self.predict(X_train)
            y_val_pred = self.predict(X_val)
            if self._loss_func == "mean_square_error":
                per_epoch_loss_train.append(self._mean_squared_error(y_train, y_train_pred))
                per_epoch_loss_val.append(self._mean_squared_error(y_val, y_val_pred))
            elif self._loss_func == "binary_cross_entropy":
                per_epoch_loss_train.append(self._binary_cross_entropy(y_train, y_train_pred))
                per_epoch_loss_val.append(self._binary_cross_entropy(y_val, y_val_pred))
            else: 
                raise ValueError(f"Unknown loss function {self._loss_func}")
            
        return per_epoch_loss_train, per_epoch_loss_val




    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        #self.forward(X) returns a tuple (y_hat, cache) where y_hat is the final output of the network (my predictions) and cache contains the intermediate Z and A values for backprop. 
        return self.forward(X)[0]

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #computes activation output from Z
        A = self._sigmoid(Z)
        #the derivative of sigmoid is A * ( 1 - A )
        #to continue backrpop apply chainrule dA * xxxx
        return dA * (A * (1 - A))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(Z, 0) 

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * (Z > 0).astype(np.float64)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        return np.mean(-1 * y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).item()

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return - y / y_hat + (1 - y) / (1 - y_hat)

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean(np.sum((y_hat - y) ** 2, axis=1)).item()

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return 2 * (y_hat - y)