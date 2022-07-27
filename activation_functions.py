import numpy as np

## Activation Functions

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