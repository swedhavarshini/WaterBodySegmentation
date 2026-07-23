import os

print("Current Working Directory:")
print(os.getcwd())

print("\nDoes dataset exist?")
print(os.path.exists("dataset"))

print("\nDoes S1Perm exist?")
print(os.path.exists("dataset/S1Perm"))

print("\nDoes JRCPerm exist?")
print(os.path.exists("dataset/JRCPerm"))

print("\nContents of dataset:")
print(os.listdir("dataset"))