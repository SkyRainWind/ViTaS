import numpy as np
import imageio

file_path = "image_obses-6.npy"
frames = np.load(file_path)
frames = frames[:, -1, :, :, :]
frames = (frames * 255).astype(np.uint8)

imageio.mimsave("image_obses.gif", frames, duration=100)