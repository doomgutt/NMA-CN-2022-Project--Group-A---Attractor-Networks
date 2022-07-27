import numpy as np


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
