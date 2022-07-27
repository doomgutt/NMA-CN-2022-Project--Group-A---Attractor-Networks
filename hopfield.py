import numpy as np

class HopfieldNetwork(object):
    def __init__(self, **kwargs):
      pass

    ## Training Step Options
    # ----------------------

    def TS_hebbian(self, training_alphabet):
        """
        Apply the hebbian training algorhithm 
        with training_alphabet as the input
        """
        m, n_units = np.shape(training_alphabet)
        self.m = m
        self.n_units = n_units
        self.training_alphabet = training_alphabet
        self.weights = np.zeros((n_units, n_units))

        # Memory lossiness warning
        if n_units*0.14 < m:
            print("The number of memory patterns to be stored is > 14%% " +
                "of the model size. This may lead to problems." +
                "ref: https://doi.org/10.3389/fncom.2016.00144")

        # Hebbian rule
        for x in training_alphabet:
            self.weights += np.outer(x, x) / m
        self.weights[np.diag_indices(n_units)] = 0
    
    def TS_storkey(self, training_alphabet):
        # TODO: check if this works?
        m, num_neurons = np.shape(training_alphabet)
        self.m = m
        self.num_neurons = num_neurons
        self.training_alphabet = training_alphabet
        self.weights = np.zeros([self.num_neurons, self.num_neurons])

        for image_vector in self.training_alphabet:
            self.weights += np.outer(image_vector, image_vector) / self.num_neurons
            net = np.dot(self.weights, image_vector)

            pre = np.outer(image_vector, net)
            post = np.outer(net, image_vector)

            self.weights -= np.add(pre, post) / self.num_neurons
        np.fill_diagonal(self.weights, 0)


    # Inference Step Options
    # ----------------------

    def IS_tanh_threshold(self, X, N, gradient=1000, threshold=1):
        """
        Run the inference step N times starting with the input X0
        The activation function is tanh(a*x + b)
        Set gradient to 1 for normal tanh
        """
        # set up empty history
        Xs = np.zeros((N, len(X)))

        for i in range(N):
            # weighted sums
            ws = np.dot(X, self.weights)

            # activation function
            X = np.tanh(gradient * (ws - threshold))

            # check if there's change from previous entry
            if i > 0:
                if self._calculate_error(Xs[i-1], X) == 0:
                    Xs = Xs[:i]
                    # print(f"quit after {i} steps: steady state reached")
                    break

            # add entry to state history
            Xs[i] = X.copy()

        self.inference_history = Xs
        return Xs
    

    ## Evaluation functions
    # ---------------------

    def energy(self):
        """sum of values >= 0 over the inference history"""
        return np.sum([self.inference_history >= 0])
    
    def time(self):
        return len(self.inference_history)

    def perf(self, test):
        """difference between chosen input"""
        return self._calculate_error(test, self.inference_history[-1])

    def best_fit(self, y_hat):
        X_predict = self.inference_history[-1]
        score = self._validate(X_predict, y_hat)
        # TODO:
        return score

    def _validate(self, X_predict, y_hat):
        '''
        Return value: (is_correct, mse_to_target)
            is_correct: 0 or 1
            mse_target: Scalar
        '''
        # find the most similar picture in the dataset
        min_error = 1e10
        min_error_idx = -1
        for y_idx in range(self.m):
            # print(X_predict, self.training_alphabet[y_idx])
            this_error = self._calculate_error(X_predict, self.training_alphabet[y_idx])
            # print(f"idx={y_idx}, error={this_error}")
            if this_error < min_error:
                min_error = this_error
                min_error_idx = y_idx
        # print(min_error, min_error_idx)
        return (min_error_idx == y_hat), self._calculate_error(X_predict, self.training_alphabet[y_hat])

    @staticmethod
    def _calculate_error(x1, x2):
        return np.sum(np.abs(x1 - x2))


class PerformanceMetric(object):
    def __init__(self):
        self.time = []
        self.energy = []
        self.is_correct = []
        self.error = []
