import os
import cv2
import matplotlib.pyplot as plt
from segment import segment

folder_path = "C:\\Users\\nicho\\OneDrive\\Desktop\\Visual Studio\\Python\\CS659\\Brain-Seg-Local\\input"
subfolder_name = 'Astrocitoma T1'
image_name = 'Astrocitoma T1_1.jpeg'
image_path = os.path.join(folder_path, subfolder_name, image_name)

fig, axes = plt.subplots(3, 4)

img = plt.imread(image_path)
if len(img.shape) > 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Original
axes[0][0].imshow(img, cmap="gray")
axes[0][0].set_title("Original Image")
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])

blur, clahe, clahe_blur, grayscaling, clahe2, clahe2_blur, min, min_blur, otsu, canny, final = segment(img)

# BLUR
axes[0][1].imshow(blur, cmap="gray")
axes[0][1].set_title("Low Pass Filter")
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])

# CLAHE
axes[0][2].imshow(clahe, cmap="gray")
axes[0][2].set_title("CLAHE")
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])

# CLAHE BLUR
axes[0][3].imshow(clahe_blur, cmap="gray")
axes[0][3].set_title("Low Pass Filter")
axes[0][3].set_xticks([])
axes[0][3].set_yticks([])

# GRAYSCALING
axes[1][0].imshow(grayscaling, cmap="gray")
axes[1][0].set_title("Grayscaling")
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])

# CLAHE 2
axes[1][1].imshow(clahe2, cmap="gray")
axes[1][1].set_title("CLAHE 2")
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])

# CLAHE 2 BLUR
axes[1][2].imshow(clahe2_blur, cmap="gray")
axes[1][2].set_title("Low Pass Filter")
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])

# MIN
axes[1][3].imshow(min, cmap="gray")
axes[1][3].set_title("Min Kernel")
axes[1][3].set_xticks([])
axes[1][3].set_yticks([])

# MIN BLUR
axes[2][0].imshow(min_blur, cmap="gray")
axes[2][0].set_title("Low Pass Filter")
axes[2][0].set_xticks([])
axes[2][0].set_yticks([])

# OTSU
axes[2][1].imshow(otsu, cmap="gray")
axes[2][1].set_title("Otsu's")
axes[2][1].set_xticks([])
axes[2][1].set_yticks([])

# CANNY
axes[2][2].imshow(canny, cmap="gray")
axes[2][2].set_title("Canny's")
axes[2][2].set_xticks([])
axes[2][2].set_yticks([])

# FINAL
axes[2][3].imshow(final, cmap="gray")
axes[2][3].set_title("Final")
axes[2][3].set_xticks([])
axes[2][3].set_yticks([])


plt.show()

