import numpy as np
import matplotlib.pyplot as plt


def show_letter(pattern, ax = None):
    if ax == None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        f.tight_layout()
    side_len = int( pattern.size ** 0.5 + 0.5)
    ax.imshow(pattern.reshape(side_len, side_len), cmap='bone_r')
    ax.set_axis_off()

def add_noise(x_, noise_level=.2):
    noise = np.random.choice(
        [1, -1], size=len(x_), p=[1-noise_level, noise_level])
    return x_ * noise