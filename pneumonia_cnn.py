from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def create_data_generators(batch_size):
    training_data_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    training_iterator = training_data_generator.flow_from_directory(
        os.path.join('data', 'train'),  # Adjust path here
        class_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size
    )

    validation_data_generator = ImageDataGenerator(rescale=1.0/255)

    validation_iterator = validation_data_generator.flow_from_directory(
        os.path.join('data', 'test'),  # Adjust path here
        class_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size
    )

    return training_iterator, validation_iterator

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(256, 256, 1)))
    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))  # Adding dropout for regularization
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.AUC()
        ]
    )

def train_model(model, training_iterator, validation_iterator, batch_size, epochs):
    model.fit(
        training_iterator,
        steps_per_epoch=training_iterator.samples/batch_size,
        epochs=epochs,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples/batch_size
    )

    # After training, evaluate on the validation set
    validation_iterator.reset()
    predictions = model.predict(validation_iterator, steps=validation_iterator.samples / BATCH_SIZE, verbose=1)
    y_true = validation_iterator.classes
    y_pred = np.argmax(predictions, axis=1)

    # Compute confusion matrix and classification report
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'])

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(class_report)

    # Save the trained model
    model.save('pneumonia_cnn.h5')

if __name__ == "__main__":
    BATCH_SIZE = 16
    EPOCHS = 10  # Increasing the number of epochs for potentially better convergence

    print("\nLoading data generators...")
    training_iterator, validation_iterator = create_data_generators(BATCH_SIZE)

    print("\nBuilding model...")
    model = build_model()
    model.summary()

    print("\nCompiling model...")
    compile_model(model)

    print("\nTraining model...")
    train_model(model, training_iterator, validation_iterator, BATCH_SIZE, EPOCHS)