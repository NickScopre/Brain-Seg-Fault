import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil as sh
from segment import segment

folder_path = "C:\\Users\\nicho\\OneDrive\\Desktop\\Visual Studio\\Python\\CS659\\Brain-Seg-Local\\input"
subfolder_name = 'Astrocitoma T1'
image_name = 'Astrocitoma T1_1.jpeg'
image_path = os.path.join(folder_path, subfolder_name, image_name)

fig, axes = plt.subplots(2, 4)

img = plt.imread(image_path)
if len(img.shape) > 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Original
axes[0][0].imshow(img, cmap="gray")
axes[0][0].set_title("Original Image")
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])

clahe, gray, min, otsu, canny, border, final = segment(img)

# CLAHE
axes[0][1].imshow(clahe, cmap="gray")
axes[0][1].set_title("CLAHE")
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])

# Grayscaling
axes[0][2].imshow(gray, cmap="gray")
axes[0][2].set_title("Gray Level Slicing")
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])


# Min Pooling
axes[0][3].imshow(min, cmap="gray")
axes[0][3].set_title("Min Pooling")
axes[0][3].set_xticks([])
axes[0][3].set_yticks([])

# Otsu's Thresholding
axes[1][0].imshow(otsu, cmap="gray")
axes[1][0].set_title("Otsu's")
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])

# Canny's Edge Detection
axes[1][1].imshow(canny, cmap="gray")
axes[1][1].set_title("Canny's")
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])

# Dialated Border
axes[1][2].imshow(border, cmap="gray")
axes[1][2].set_title("Border")
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])

# Final
axes[1][3].imshow(final, cmap="gray")
axes[1][3].set_title("Final Output")
axes[1][3].set_xticks([])
axes[1][3].set_yticks([])


plt.show()

