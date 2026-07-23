from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

def build_model():

    inputs = Input(shape=(128,128,1))

    x = Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)

    x = UpSampling2D((2,2))(x)

    outputs = Conv2D(1,(1,1),activation='sigmoid')(x)

    model = Model(inputs,outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()