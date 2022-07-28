import numpy as np
import matplotlib.pyplot as plt
# import hopfield


def show_letter(pattern, ax = None):
    if ax == None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        f.tight_layout()
    side_len = int(pattern.size ** 0.5)
    trimmed = pattern[:side_len*side_len]
    ax.imshow(trimmed.reshape(side_len, side_len), cmap='bone', vmin=-1, vmax=1)
    ax.set_axis_off()
    # how do we talk about

def add_noise(x_, noise_level=.2):
    noise = np.random.uniform(-1, 1, len(x_))
    noise = noise * np.random.choice([.0, 1.0], size=len(x_), p=[1-noise_level, noise_level])
    noise[noise==.0] = 1
    return x_ * noise

def old_add_noise(x_, noise_level=.2):
    noise = np.random.choice(
        [1, -1], size=len(x_), p=[1-noise_level, noise_level])
    return x_ * noise

def noisify_dataset(dataset, iterations=10, noise=.1):
    noized_dataset = []
    for img in dataset:
        for _ in range(iterations):
            noized_dataset.append(add_noise(img, noise))
    return np.array(noized_dataset)



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