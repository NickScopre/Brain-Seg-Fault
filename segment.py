import cv2
import numpy as np

def segment(img):
    
    # Low Pass Filter to reduce noise
    img1 = cv2.GaussianBlur(img, (11, 11), 0)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) (was 2.0)
    clahe_inst = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(15, 15))
    clahe = clahe_inst.apply(img1)

    # Low Pass Filter
    clahe_blur = cv2.GaussianBlur(clahe, (51, 51), 0)

    # Gray Level Slicing
    intensity_ranges = [
        (0, 45, 0),
        (46, 55, 150),
        (56, 75, 200),
        (76, 99, 150),
        (100, 255, 0)
    ]
    
    # Apply gray level slicing for each intensity range mapping
    grayscaling = np.copy(clahe_blur)
    for intensity_range in intensity_ranges:
        min_intensity, max_intensity, slice_intensity = intensity_range
        grayscaling[(clahe_blur >= min_intensity) & (clahe_blur <= max_intensity)] = slice_intensity


    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe2 = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(5, 5))
    clahe2 = clahe2.apply(grayscaling)

    # Second Low Pass Filter
    clahe2_blur = cv2.GaussianBlur(clahe2, (25, 25), 0)

    # Minimum Kernel
    min_range = int(np.ceil(max(img.shape)*0.04))
    min = cv2.erode(clahe2_blur, np.ones((min_range, min_range), np.uint8))

    # Third Low Pass Filter 
    img6 = cv2.GaussianBlur(min, (201, 201), 0)

    # Clip to Ensure in Range
    img7 = np.clip(img6, 0, 255).astype(np.uint8)

    # Apply Otsu's Algorithm of Thresholding
    _, otsu = cv2.threshold(img7, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny's Edge Detection
    canny = cv2.Canny(otsu, 100, 200)

    # Border
    border = cv2.dilate(canny, np.ones((3, 3), np.uint8))

    # Darken original image
    final = np.clip(img.astype(int)-50, 0, 255).astype(np.uint8) 

    #Outline
    map1 = (otsu == 255) # Create mask for Otsu's Output
    map2 = (border == 255) # Create mask for Canny's Border
    final[map1] = np.clip(final[map1].astype(int) + 25, 0, 255).astype(np.uint8)
    final[map2] = np.clip(final[map2].astype(int) + 75, 0, 255).astype(np.uint8) 

    return img1, clahe, clahe_blur, grayscaling, clahe2, clahe2_blur, min, img6, otsu, border, final