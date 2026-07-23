import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# ------------------------
# Load Dataset
# ------------------------

X = np.load("dataset/X.npy")
Y = np.load("dataset/Y.npy")

# ------------------------
# Train / Validation Split
# ------------------------

_, X_test, _, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ------------------------
# Load Best Model
# ------------------------

model = load_model("water_segmentation_model.keras")

# ------------------------
# Evaluate
# ------------------------

loss, accuracy = model.evaluate(X_test, Y_test)

predictions = model.predict(X_test)

predictions = (predictions > 0.5).astype(np.float32)

def iou_score(y_true, y_pred):

    intersection = np.logical_and(y_true, y_pred).sum()

    union = np.logical_or(y_true, y_pred).sum()

    return intersection / (union + 1e-7)

def dice_score(y_true, y_pred):

    intersection = np.logical_and(y_true, y_pred).sum()

    return (2 * intersection) / (y_true.sum() + y_pred.sum() + 1e-7)
iou = iou_score(Y_test, predictions)

dice = dice_score(Y_test, predictions)

print(f"IoU Score  : {iou:.4f}")

print(f"Dice Score : {dice:.4f}")

print("\n==========================")
print("Evaluation Results")
print("==========================")

print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy*100:.2f}%")