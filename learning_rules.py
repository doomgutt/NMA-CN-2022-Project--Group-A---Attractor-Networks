import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)


def hebbian(training_set):
    """
    Apply the hebbian training algorhithm 
    with training_set as the input
    """
    ts_size, hn_size = np.shape(training_set)
    weights = np.zeros((hn_size, hn_size))

    # Memory lossiness warning
    if hn_size*0.14 < ts_size:
        logging.info("The number of memory patterns to be" + 
            "stored is > 14%% of the model size. This may lead " +
            "to problems. \nref: https://doi.org/10.3389/fncom.2016.00144")

    # Hebbian rule
    for x in training_set:
        weights += np.outer(x, x) / ts_size
    weights[np.diag_indices(hn_size)] = 0

    return weights



def storkey(training_set):
    # TODO: check if this works?
    ts_size, hn_size = np.shape(training_set)
    weights = np.zeros([hn_size, hn_size])

    ## image vector is x
    # self.X  is training_set

    for x in training_set:
        weights += np.outer(x, x) / hn_size
        net = np.dot(weights, x)

        pre = np.outer(x, net)
        post = np.outer(net, x)

        weights -= np.add(pre, post) / hn_size
    np.fill_diagonal(weights, 0)

    return weights

# ----------------------------------------
dictionary = {
  "hebbian": hebbian,
  "storkey": storkey
}


#########################################
## BACKUPS

# def TS_hebbian(self, training_alphabet):
#     """
#     Apply the hebbian training algorhithm 
#     with training_alphabet as the input
#     """
#     m, n_units = np.shape(training_alphabet)
#     self.m = m
#     self.n_units = n_units
#     self.training_alphabet = training_alphabet
#     self.weights = np.zeros((n_units, n_units))

#     # Memory lossiness warning
#     if n_units*0.14 < m:
#         print("The number of memory patterns to be stored is > 14%% " +
#             "of the model size. This may lead to problems." +
#             "ref: https://doi.org/10.3389/fncom.2016.00144")

#     # Hebbian rule
#     for x in training_alphabet:
#         self.weights += np.outer(x, x) / m
#     self.weights[np.diag_indices(n_units)] = 0

# def TS_storkey(self, training_alphabet):
#     # TODO: check if this works?
#     m, n_units = np.shape(training_alphabet)
#     self.m = m
#     self.n_units = n_units
#     self.training_alphabet = training_alphabet
#     self.weights = np.zeros([n_units, n_units])

#     ## image vector is x
#     # self.X  is training_alphabet

#     for x in training_alphabet:
#         self.weights += np.outer(x, x) / self.n_units
#         net = np.dot(self.weights, x)

#         pre = np.outer(x, net)
#         post = np.outer(net, x)

#         self.weights -= np.add(pre, post) / self.n_units
#     np.fill_diagonal(self.weights, 0)















#########################################



#storkey learning rule

def storkeyyyy(self):
    self.W = np.zeros([self.num_neurons, self.num_neurons])

    for image_vector, _ in self.train_dataset:
        self.W += np.outer(image_vector, image_vector) / self.num_neurons
        net = np.dot(self.W, image_vector)

        pre = np.outer(image_vector, net)
        post = np.outer(net, image_vector)

        self.W -= np.add(pre, post) / self.num_neurons
    np.fill_diagonal(self.W, 0)


def extended_storkey_update(x, weights):
    """
    Create an Op that performs a step of the Extended
    Storkey Learning Rule.
    Args:
        sample: a 1-D x Tensor of dtype tf.bool.
        weights: the weight matrix to update.
    Returns:
        An Op that updates the weights based on the sample.
    """
    scale = 1 / int(weights.get_shape()[0])
    numerics = 2*tf.cast(sample, weights.dtype) - 1
    row_sample = tf.expand_dims(numerics, axis=0)
    row_h = tf.matmul(row_sample, weights)

    pos_term = (tf.matmul(tf.transpose(row_sample), row_sample) +
                tf.matmul(tf.transpose(row_h), row_h))
    neg_term = (tf.matmul(tf.transpose(row_sample), row_h) +
                tf.matmul(tf.transpose(row_h), row_sample))
    return tf.assign_add(weights, scale * (pos_term - neg_term))



#Pseudo-Inverse Rule (implemented in C++)
# if (method == LearningMethod::PSEUDO_INVERSE)
# {
#     arma::mat w(N, N);
#     int m = samples.size(); // št. učnih vzorcev
#     for (int i = 0; i != N; i++)
#     {
#         for (int j = 0; j != N; j++)
#         {
#             w(i, j) = 0.0;
#             for (int v = 0; v != m; v++)
#             {
#                 for (int u = 0; u != m; u++)
#                 {
#                     w(i, j) += samples[v][i] * (1.0 / Q(u, v, samples)) * samples[u][j];
#                 }
#             }
#             w(i, j) = w(i, j) / N;
#         }
#     }
#     std::cout << w << endl;

#     // prekopiraj iz Armadillo mat objekta v 2D array
#     for (int i = 0; i != N; i++)
#     {
#         for (int j = 0; j != N; j++)
#         {
#             W[i][j] = w(i, j);
#         }
#     }
# }
