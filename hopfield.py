import numpy as np
import activation_functions
import learning_rules
import utilities as uti
import energy_functions
import logging

logging.basicConfig(level=logging.INFO)

class HopfieldNetwork(object):
    def __init__(self, **kwargs):
        pass

    def run(self, training_set, iterations, params=(1,0),
            lr='hebbian', af='sync_tanh',
            pe_fn = "pe_lin", se_fn = 'se_lin',
            n_test_samples=9, noise_level=.0,
            noize_dataset=True, stop_error=1e-5,
            performance_threshold=0, print_info=True):

        if noize_dataset:
            noized_trainig_set = uti.noisify_dataset(training_set)
            self.training_step(noized_trainig_set, lr)
            self.training_set = training_set
        else:
            self.training_step(training_set, lr)

        pm = PerformanceMetric()
        correctness_list = []
        for i in range(n_test_samples):
            # idx = np.random.randint(0, n_images)
            idx = i % len(training_set)

            x_test = training_set[idx].copy()
            x_test = uti.add_noise(x_test, noise_level=noise_level)

            inf_hist = self.inference_step(x_test, iterations, af=af, params=params, stop_error=stop_error)

            is_correct, error = self._validate(inf_hist[-1], idx)
            pm.add(is_correct, error, self.time(), 
                self.process_energy(pe_fn), self.sequence_energy(se_fn))
            
            correctness_list.append(is_correct)

            if print_info:
                ax = uti.plt.subplot(3, n_test_samples, i+1)
                uti.show_letter(training_set[idx], ax)
                ax = uti.plt.subplot(3, n_test_samples, n_test_samples + i+1)
                uti.show_letter(x_test, ax)
                ax = uti.plt.subplot(3, n_test_samples, 2*n_test_samples + i+1)
                uti.show_letter(inf_hist[-1], ax)

            # print(idx, is_correct, error)
            # print(len(inf_hist))
            # print(inf_hist)
            # print(inf_hist[-1])
            # print(x_test.dtype, inf_hist[-1].dtype)
        if print_info: print(correctness_list)
        return pm


    ## Training Step
    # --------------
    def training_step(self, training_set, learning_rule="hebbian"):
        self.lr_dict = learning_rules.dictionary
        self.training_set = training_set
        self.size = training_set.shape[1]
        self.weights = self.lr_dict[learning_rule](training_set)

    # Inference Step
    # --------------
    def inference_step(self, X, iterations, params=(1, 0), af="sync_tanh",
                       stop_step=10, stop_error=1e-5):
        self.af_dict = activation_functions.dictionary
        X = X.astype("float")
        # print(self.af_dict)
        Xs = np.zeros((iterations, len(X)))
        for i in range(iterations):
            X = self.af_dict[af](X, self.weights, *params)
            if i >= stop_step:
                if self._calculate_error(Xs[i-stop_step], X) < stop_error:
                    Xs = Xs[:i]
                    logging.info(f"quit after {i} steps: steady state reached")
                    break
            Xs[i] = X.copy()
        self.inference_history = Xs
        return Xs
     

    ## Evaluation functions
    # ---------------------

    def process_energy(self, pe_fn):
        """ energy of the whole process 
        "pe_lin"  : process_energy_lin,
        "pe_abs"  : process_energy_abs,
        "pe_relu" : process_energy_relu,
        """
        self.pe_dict = energy_functions.process_energy_dictionary
        return self.pe_dict[pe_fn](self.inference_history)

    def sequence_energy(self, se_fn):
        """ energy sequence for every consecutive state 
        "se_lin"  : state_energy_bio_lin,
        "se_abs"  : state_energy_bio_abs,
        "se_relu" : state_energy_bio_relu,
        """
        self.se_dict = energy_functions.state_energy_dictionary
        energy_history = np.zeros(len(self.inference_history))
        for i, X in enumerate(self.inference_history):
            energy_history[i] = self.se_dict[se_fn](X)
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
        # print("X_predict", X_predict)
        min_error = 1e10
        min_error_idx = -1
        for y_idx in range(self.training_set.shape[0]):
            # print(f"y_idx={y_idx}", self.training_set[y_idx])
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
        return np.sum(np.abs(x1 - x2)) / 2 / x1.size


class PerformanceMetric(object):
    def __init__(self):
        self.is_correct = []
        self.error = []
        self.timestamp = []
        self.process_energy = []
        self.sequence_energy = []

    def add(self, _is_correct, _error, _timestamp, _process_energy, _sequence_energy):
        self.is_correct.append(_is_correct)
        self.error.append(_error)
        self.timestamp.append(_timestamp)
        self.process_energy.append(_process_energy)
        self.sequence_energy.append(_sequence_energy)

    def __str__(self):
        ret_str = "{"
        ret_str += "is_correct:" + str(self.is_correct) + ","
        ret_str += "error:" + str(self.error) + ","
        ret_str += "timestamp:" + str(self.timestamp) + ","
        ret_str += "process_energy:" + str(self.process_energy) + ","
        ret_str += "sequence_energy:" + str(self.sequence_energy) + ","
        ret_str += "}"
        return ret_str