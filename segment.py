import cv2
import numpy as np

def segment(img):
    
    # Low Pass Filter to reduce noise
    img1 = cv2.GaussianBlur(img, (11, 11), 0)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img2 = clahe.apply(img1)

    # Gray Level Slicing
    intensity_ranges = [
        (0, 50, 0),
        (100, 255, 0)
    ]

    # Apply gray level slicing for each intensity range mapping
    grayscaling = np.copy(img2)
    for intensity_range in intensity_ranges:
        min_intensity, max_intensity, slice_intensity = intensity_range
        grayscaling[(img2 >= min_intensity) & (img2 <= max_intensity)] = slice_intensity


    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(3, 3))
    clahe = clahe.apply(grayscaling)

    # Minimum Kernel
    min_range = int(np.ceil(max(img.shape)*0.03))
    min = cv2.erode(clahe, np.ones((min_range, min_range), np.uint8))

    img6 = cv2.GaussianBlur(min, (11, 11), 0)

    # Clip to Ensure in Range
    img7 = np.clip(img6, 0, 255).astype(np.uint8)

    # Apply Otsu's Algorithm of Thresholding
    _, otsu = cv2.threshold(img7, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny's Edge Detection
    canny = cv2.Canny(min, 100, 200)

    # somethin
    border = cv2.dilate(canny, np.ones((3, 3), np.uint8))

    final = np.clip(img.astype(int)-50, 0, 255).astype(np.uint8) # Darken original image
    map1 = (otsu == 255) # Create mask for Otsu's Output
    map2 = (border == 255) # Create mask for Canny's Border
    final[map1] = np.clip(final[map1].astype(int) + 50, 0, 255).astype(np.uint8)
    final[map2] = np.clip(final[map2].astype(int) + 75, 0, 255).astype(np.uint8) 

    return clahe, grayscaling, min, otsu, canny, border, final