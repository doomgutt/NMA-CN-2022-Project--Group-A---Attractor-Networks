Latest updates:

new functions:
- HopfieldNetwork.inference_step(X_input, iterations, "af_string")
    - use instead of IS_tanh_sync()
- sync_tanh(X, iterations): updating all at once
- async_tanh(X, iterations): update 1 at a time randomly
- async_n_tanh(X, iterations): update n at a time randomly (equivalent to sync_tan when n = len(X))

problems:
- validate doesn't work as it should

bug fixes:
- update functions had X.dot(weights) instead of weights.dot(X)
