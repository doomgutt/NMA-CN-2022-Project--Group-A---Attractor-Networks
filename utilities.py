import numpy as np
import matplotlib.pyplot as plt
# import hopfield


def show_letter(pattern, ax = None):
    if ax == None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        f.tight_layout()
    side_len = int( pattern.size ** 0.5 + 0.5)
    ax.imshow(pattern.reshape(side_len, side_len), cmap='bone_r')
    ax.set_axis_off()
    # how do we talk about

def add_noise(x_, noise_level=.2):
    noise = np.random.choice(
        [1, -1], size=len(x_), p=[1-noise_level, noise_level])
    return x_ * noise





# def run_inference(dataset, lr, af, iterations, n_test_samples=9, noise_level=.0):
#     hop_net = hopfield.HopfieldNetwork()
#     hop_net.training_step(dataset, lr)

#     n_images = len(dataset)
#     for i in range(n_test_samples):
#         # add noise
#         # idx = np.random.randint(0, n_images)
#         idx = i % n_images

#         x_test = dataset[idx].copy()
#         x_test = add_noise(x_test, noise_level=noise_level)

#         Xs = hop_net.inference_step(x_test, iterations, af)

#         is_correct, error = hop_net._validate(x_test, idx)
#         print(idx, is_correct, error)
#         print(len(Xs))

#         ax = plt.subplot(2, n_test_samples, i+1)
#         show_letter(x_test, ax)
#         ax = plt.subplot(2, n_test_samples, n_test_samples + i+1)
#         show_letter(Xs[-1], ax)
#         # print(x_test.dtype, Xs[-1].dtype)