import numpy as np


def sync_tanh(X, weights, gradient=1, threshold=0, size_norm=None):
    ws = weights.dot(X)
    return tanh_in(ws, gradient, threshold, size_norm)

def async_tanh(X, weights, gradient=1, threshold=0, size_norm=None):
    i = np.random.randint(len(X))
    ws_i = np.sum(X * weights[i])
    X[i] = tanh_in(ws_i, gradient, threshold, size_norm)
    return X

def async_n_tanh(X, weights, n_choices=None, gradient=1, threshold=0, size_norm=None):
    if n_choices == None:
        n_choices = len(X)//2
    elif n_choices >= len(X):
        n_choices = len(X)

    idx = np.random.choice(len(X), size=n_choices, replace=False)
    idx_weights = np.zeros((len(idx), 2))

    for n, i in enumerate(idx):
        idx_weights[n][0] = i
        idx_weights[n][1] = np.sum(X * weights[i])
    for i_w in idx_weights:
        X[int(i_w[0])] = tanh_in(i_w[1], gradient, threshold, size_norm)
    return X

def tanh_in(x, gradient, threshold, size_norm=None):
    c = 1
    if size_norm:
        c = 1 / size_norm
    return np.tanh(c * gradient * (x - threshold))

# ----------------------------------------
dictionary = {
    "sync_tanh": sync_tanh,
    "async_tanh": async_tanh,
    "async_n_tanh": async_n_tanh
}


#########################################
## Backups

    # def IS_sync_tanh(self, X, N, gradient=1, threshold=0, step_check=1):
    #     """
    #     Run the inference step N times starting with the input X0
    #     The activation function is tanh(a*x + b)
    #     Set gradient to 1 for normal tanh
    #     """
    #     Xs = np.zeros((N, len(X)))
    #     for i in range(N):
    #         ws = np.dot(X, self.weights)
    #         X = np.tanh(gradient * (ws - threshold))
    #         if i >= step_check:
    #             if self._calculate_error(Xs[i-step_check], X) == 0:
    #                 Xs = Xs[:i]
    #                 # print(f"quit after {i} steps: steady state reached")
    #                 break
    #         Xs[i] = X.copy()
    #     self.inference_history = Xs
    #     return Xs

    # def IS_async_tanh(self, X, N, gradient=1, threshold=0, step_check=10):
    #     """ async tanh """
    #     Xs = np.zeros((N, len(X)))
    #     for i in range(N):
    #         choice = np.random.randint(len(X))
    #         choice_w_in = np.sum(X[choice] * self.weights)
    #         X[choice] = np.tanh(choice_w_in)
    #         if i >= step_check:
    #             if self._calculate_error(Xs[i-step_check], X) == 0:
    #                 Xs = Xs[:i]
    #                 # print(f"quit after {i} steps: steady state reached")
    #                 break
    #         Xs[i] = X.copy()
    #     self.inference_history = Xs
    #     return Xs




















#########################################




def ReLU(ws):
    data = [max(0,value) for value in ws]
    return np.array(data, dtype=float)

## for a single element
def leakyRelu(ws):
    if ws>0:
        return ws
    else:
        return 0.01*ws

## for a matrix
def leaky_relu(ws):
    alpha = 0.1
    return np.maximum(alpha*ws, ws)