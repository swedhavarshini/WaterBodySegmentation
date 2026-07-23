import os
import cv2
import numpy as np
import tifffile as tiff

# ======================================================
# Dataset Paths
# ======================================================
IMAGE_DIR = "dataset/S1Perm"
MASK_DIR = "dataset/JRCPerm"

IMG_SIZE = (128, 128)

# ======================================================
# Find Matching Files
# ======================================================

image_files = set(f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif"))
mask_files = set(f for f in os.listdir(MASK_DIR) if f.endswith(".tif"))

common_files = sorted(image_files.intersection(mask_files))

print("=" * 60)
print("Dataset Information")
print("=" * 60)

print(f"Images Found       : {len(image_files)}")
print(f"Masks Found        : {len(mask_files)}")
print(f"Matching Pairs     : {len(common_files)}")
print(f"Images without Mask: {len(image_files - mask_files)}")
print(f"Masks without Image: {len(mask_files - image_files)}")

print("=" * 60)

images = []
masks = []

# ======================================================
# Preprocess Every Matching Pair
# ======================================================

for i, filename in enumerate(common_files):

    print(f"Processing {i+1}/{len(common_files)} : {filename}")

    try:
        image_path = os.path.join(IMAGE_DIR, filename)
        mask_path = os.path.join(MASK_DIR, filename)

        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)

        image = image[0]

        image = cv2.resize(image, IMG_SIZE)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        # Remove NaN values
        image = np.nan_to_num(image, nan=0.0)

# Normalize
        image = image.astype(np.float32)

        if image.max() > 0:
            image = image / image.max()

        # Remove NaN values
        mask = np.nan_to_num(mask, nan=-1)

# Convert (-1,0,1) → (0,0.5,1)
        mask = (mask + 1) / 2

# Convert to binary
        mask = (mask > 0.5).astype(np.float32)

        image = image.reshape(128, 128, 1)
        mask = mask.reshape(128, 128, 1)

        images.append(image)
        masks.append(mask)

    except Exception as e:
        print(f"Error processing {filename}")
        print(e)

# ======================================================
# Convert to NumPy Arrays
# ======================================================

X = np.array(images)
Y = np.array(masks)

np.save("dataset/X.npy", X)
np.save("dataset/Y.npy", Y)

X = np.load("dataset/X.npy")
Y = np.load("dataset/Y.npy")

# ======================================================
# Dataset Summary
# ======================================================

print("\n")
print("=" * 60)
print("Preprocessing Completed")
print("=" * 60)

print("X Shape :", X.shape)
print("Y Shape :", Y.shape)

print()

print("Image Min :", X.min())
print("Image Max :", X.max())

print()

print("Mask Min :", Y.min())
print("Mask Max :", Y.max())

print()

print("Unique Mask Values :", np.unique(Y))