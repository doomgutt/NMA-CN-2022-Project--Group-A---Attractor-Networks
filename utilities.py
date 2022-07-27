import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_all_letter_dataset(img, new_size = 64):
    pixel_i = 115
    pixel_j = 110
    all_letter_final = np.empty((9 * 9, new_size, new_size), dtype="int8")
    for i in range(9):
        for j in range(9):
            # print(i, j)
            i_offset = 0
            if i >= 3:
                i_offset = 40
            if i >= 6:
                i_offset = 80

            subimg = img[i_offset + pixel_i * i : i_offset +pixel_i * (i+1),
                         pixel_j * j : pixel_j * (j+1),
                         :4].astype("uint8")
            new_img = Image.fromarray(subimg)

            new_img = new_img.resize((new_size, new_size))
            new_arr = np.asarray(new_img)
            # print(new_arr[20:30, 20:30, 3])
            new_arr = (new_arr[:, :, 3] > 100) * 2 - 1
            # all_letter_final[i * 9 + j] = new_arr[:, :, 0]
            all_letter_final[i*9+j] = new_arr[:]
    all_letter_final = all_letter_final.reshape(all_letter_final.shape[0], -1)
    return all_letter_final

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