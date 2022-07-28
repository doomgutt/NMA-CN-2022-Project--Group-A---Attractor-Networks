import numpy as np
import activation_functions
import learning_rules
import utilities as uti
import energy_functions

class HopfieldNetwork(object):
    def __init__(self, **kwargs):
        pass

    def run(self, dataset, lr, af, iterations, n_test_samples=9, noise_level=.0):
        self.training_step(dataset, lr)

        n_images = len(dataset)
        for i in range(n_test_samples):
            # add noise
            # idx = np.random.randint(0, n_images)
            idx = i % n_images

            x_test = dataset[idx].copy()
            x_test = uti.add_noise(x_test, noise_level=noise_level)

            Xs = self.inference_step(x_test, iterations, af)

            is_correct, error = self._validate(x_test, idx)
            print(idx, is_correct, error)
            print(len(Xs))

            ax = uti.plt.subplot(2, n_test_samples, i+1)
            uti.show_letter(x_test, ax)
            ax = uti.plt.subplot(2, n_test_samples, n_test_samples + i+1)
            uti.show_letter(Xs[-1], ax)
            # print(x_test.dtype, Xs[-1].dtype)

    ## Training Step
    # --------------
    def training_step(self, training_set, learning_rule="hebbian"):
        self.lr_dict = learning_rules.dictionary
        self.training_set = training_set
        self.size = training_set.shape[1]
        self.weights = self.lr_dict[learning_rule](training_set)

    # Inference Step
    # --------------
    def inference_step(self, X, iterations, af="sync_tanh", step_check=10):
        self.af_dict = activation_functions.dictionary
        X = X.astype("float")
        # print(self.af_dict)
        Xs = np.zeros((iterations, len(X)))
        for i in range(iterations):
            X = self.af_dict[af](X, self.weights)
            if i >= step_check:
                if self._calculate_error(Xs[i-step_check], X) == 0:
                    Xs = Xs[:i]
                    # print(f"quit after {i} steps: steady state reached")
                    break
            Xs[i] = X.copy()
        self.inference_history = Xs
        return Xs
     

    ## Evaluation functions
    # ---------------------

    def process_energy(self, energy_fn):
        """ energy of the whole process """
        self.pe_dict = energy_functions.process_energy_dictionary
        return self.pe_dict[energy_fn](self.inference_history)

    def sequence_energy(self, energy_fn):
        """ energy sequence for every consecutive state """
        self.se_dict = energy_functions.state_energy_dictionary
        energy_history = np.zeros(len(self.inference_history))
        for i, X in enumerate(self.inference_history):
            energy_history[i] = self.pe_dict[energy_fn](X)
        return energy_history

    # Time
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
        for y_idx in range(self.size):
            # print(X_predict, self.training_set[y_idx])
            this_error = self._calculate_error(X_predict, self.training_set[y_idx])
            # print(f"idx={y_idx}, error={this_error}")
            if this_error < min_error:
                min_error = this_error
                min_error_idx = y_idx
        # print(min_error, min_error_idx)
        return (min_error_idx == y_hat), self._calculate_error(X_predict, self.training_set[y_hat])

    @staticmethod
    def _calculate_error(x1, x2):
        return np.sum(np.abs(x1 - x2))


class PerformanceMetric(object):
    def __init__(self):
        self.time = []
        self.energy = []
        self.is_correct = []
        self.error = []
