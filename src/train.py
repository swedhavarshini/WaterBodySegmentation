import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint

from build_model import build_model

# --------------------------------
# Load Dataset
# --------------------------------

X = np.load("dataset/X.npy")
Y = np.load("dataset/Y.npy")

print("Dataset Loaded")

print("X:",X.shape)
print("Y:",Y.shape)

# --------------------------------
# Split Dataset
# --------------------------------

X_train,X_val,Y_train,Y_val = train_test_split(

    X,
    Y,

    test_size=0.2,

    random_state=42

)

print()

print("Training Images :",len(X_train))
print("Validation Images:",len(X_val))

# --------------------------------
# Build CNN
# --------------------------------

model = build_model()

# --------------------------------
# Save Best Model
# --------------------------------

checkpoint = ModelCheckpoint(

    "water_segmentation_model.keras",

    save_best_only=True,

    monitor="val_loss"

)

# --------------------------------
# Train
# --------------------------------

history = model.fit(

    X_train,

    Y_train,

    validation_data=(X_val,Y_val),

    epochs=20,

    batch_size=16,

    callbacks=[checkpoint]

)

print("Training Completed")