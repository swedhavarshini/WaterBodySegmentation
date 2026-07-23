import os
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tensorflow.keras.models import load_model

# ==========================================================
# Load Trained Model
# ==========================================================

model = load_model("water_segmentation_model.keras")

print("✅ Model Loaded Successfully")

# ==========================================================
# Dataset Folder
# ==========================================================

IMAGE_DIR = "dataset/S1Perm"
MASK_DIR = "dataset/JRCPerm"

# ==========================================================
# Select One Test Image
# ==========================================================

filename = sorted(os.listdir(IMAGE_DIR))[0]

print("Testing Image :", filename)

# ==========================================================
# Read SAR Image
# ==========================================================

image = tiff.imread(os.path.join(IMAGE_DIR, filename))

# Use VV Band

image = image[0]

original_image = image.copy()

# ==========================================================
# Read Ground Truth Mask
# ==========================================================

mask = tiff.imread(os.path.join(MASK_DIR, filename))

# ==========================================================
# Resize
# ==========================================================

image = cv2.resize(image, (128,128))

mask = cv2.resize(mask, (128,128))

# ==========================================================
# Normalize
# ==========================================================

image = image.astype(np.float32)

image = image / image.max()

# ==========================================================
# CNN Input Shape
# ==========================================================

input_image = image.reshape(1,128,128,1)

# ==========================================================
# Prediction
# ==========================================================

prediction = model.predict(input_image)[0]

prediction = (prediction > 0.5).astype(np.uint8)

# ==========================================================
# Calculate Water and Land Percentage
# ==========================================================

water_pixels = np.sum(prediction == 1)
land_pixels = np.sum(prediction == 0)

total_pixels = prediction.size

water_percentage = (water_pixels / total_pixels) * 100
land_percentage = (land_pixels / total_pixels) * 100

print(f"Water Area : {water_percentage:.2f}%")
print(f"Land Area  : {land_percentage:.2f}%")

# ==========================================================
# Create Colored Prediction
# ==========================================================

colored_prediction = np.zeros((128, 128, 3), dtype=np.uint8)

# Water = Blue (prediction == 1)
colored_prediction[prediction.squeeze() == 1] = [0, 0, 255]

# Land = Brown (prediction == 0)
colored_prediction[prediction.squeeze() == 0] = [139, 69, 19]

# ==========================================================
# Convert Original Image to RGB
# ==========================================================

image_uint8 = (image * 255).astype(np.uint8)

original_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

# ==========================================================
# Overlay
# ==========================================================

overlay = cv2.addWeighted(
    original_rgb,
    0.8,
    colored_prediction,
    0.2,
    0
)


# ==========================================================
# Display
# ==========================================================

plt.figure(figsize=(20,14))

# -----------------------
# Original Image
# -----------------------
plt.subplot(3,3,1)
plt.imshow(image, cmap="gray")
plt.title("Original SAR Image")
plt.axis("off")

# -----------------------
# Ground Truth
# -----------------------
plt.subplot(3,3,2)
plt.imshow(mask, cmap="Blues")
plt.title("Ground Truth Mask")
plt.axis("off")

# -----------------------
# Prediction
# -----------------------
plt.subplot(3,3,3)
plt.imshow(colored_prediction)
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(3,3,4)

plt.imshow(prediction.squeeze(), cmap="gray")

plt.title("Predicted Mask (Binary)")

plt.axis("off")

# -----------------------
# Overlay
# -----------------------
plt.subplot(3,3,5)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis("off")

plt.subplot(3,3,6)

plt.axis("off")

plt.title("Water - Land Statistics")



plt.text(
    0.05,0.90,
    f"Water Area : {water_percentage:.2f} %",
    fontsize=13,
    weight="bold",
    color="blue"
)

plt.text(
    0.05,0.72,
    f"Land Area : {land_percentage:.2f} %",
    fontsize=13,
    weight="bold",
    color="saddlebrown"
)

plt.text(
    0.05,0.54,
    f"Total Pixels : {total_pixels}",
    fontsize=12
)

plt.subplot(3,3,7)

# -----------------------
# Histogram
# -----------------------
plt.subplot(3,3,7)

plt.hist(
    prediction.flatten(),
    bins=[-0.5,0.5,1.5],
    color="gray",
    edgecolor="black"
)

plt.xticks([0,1],["Land","Water"])
plt.xlabel("Pixel Class")
plt.ylabel("Pixel Count")
plt.title("Histogram of Predicted Mask")

plt.title("Histogram of Original SAR Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.grid(alpha=0.3)

plt.grid(alpha=0.3)

plt.title("Histogram of Original SAR Image")

plt.xlabel("Pixel Value")

plt.ylabel("Pixel Count")

plt.subplot(3,3,8)

plt.pie(
    [water_percentage, land_percentage],
    labels=["Water","Land"],
    colors=["royalblue","saddlebrown"],
    autopct="%1.1f%%",
    startangle=90,
    explode=(0.03,0)
)

plt.title("Water vs Land Area (%)")

plt.subplot(3,3,9)

plt.axis("off")

legend_elements = [

    Patch(facecolor="blue", label="Water"),

    Patch(facecolor="saddlebrown", label="Land")

]

plt.legend(
    handles=legend_elements,
    loc="center",
    fontsize=12
)

plt.title("Legend")

plt.subplots_adjust(
    hspace=0.35,
    wspace=0.25
)
plt.show()