import os
import numpy as np
import matplotlib.pyplot as plt
import shutil as sh
from segment import segment

# Change renamed to False if input image names are inconsistent, but folder names indicate diagnosis
renamed = True
original_folder = "input"

# Check if the input folders exists, if it doesn't raise an error
if not os.path.exists(original_folder):
    print(f"Error: No \"{original_folder}\" folder found in local directory.")
    exit(1)
    # Maybe check if there's at least one non-folder element in the input folder here


if not renamed:
    #Iterate through each folder in input
    for subdir in os.listdir(original_folder):
        count = 1
        # Iterate through each item in that folder
        for image_name in os.listdir(os.path.join(original_folder, subdir)):
            # Rename each image to the consistent form of Diagnosis_number
            _, file_extension = os.path.splitext(image_name)
            name = subdir+"_"+str(count)+file_extension
            count += 1
            os.rename(os.path.join(original_folder, subdir, image_name), (os.path.join(original_folder, subdir, name)))


# Create the "output" folder if it doesn't exist already
segmented_folder = "output"
if os.path.exists(segmented_folder):
    sh.rmtree(segmented_folder)
else:
    os.makedirs(segmented_folder)


# Copy Subfolders in input to output
dirs = []
for subdir in os.listdir(original_folder):
    if(os.path.isdir(os.path.join(original_folder, subdir))):
        os.makedirs(os.path.join(segmented_folder, subdir), exist_ok=True)

#Iterate through each folder in input
for subdir in os.listdir(original_folder):
    count = 1
    # Iterate through each item in that folder
    for image_name in os.listdir(os.path.join(original_folder, subdir)):
        # Transform the image
        img = plt.imread(os.path.join(original_folder, subdir, image_name))
        img_t = segment(img)

        # Write the transformed image to output folder
        _, file_extension = os.path.splitext(image_name)
        #name = subdir+"_"+str(count)+file_extension
        plt.imsave(os.path.join(segmented_folder, subdir, image_name), img_t, cmap="gray")
        count += 1