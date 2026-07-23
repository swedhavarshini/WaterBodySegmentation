import os

image_dir = "dataset/S1Perm"
mask_dir = "dataset/JRCPerm"

print("Image Directory:", image_dir)
print("Mask Directory :", mask_dir)

print("\nFirst 5 files in S1Perm:")
print(os.listdir(image_dir)[:5])

print("\nFirst 5 files in JRCPerm:")
print(os.listdir(mask_dir)[:5])