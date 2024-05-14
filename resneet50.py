import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from PIL import Image
import random
from sklearn.utils.class_weight import compute_class_weight

random.seed(0)
np.random.seed(0)

datasize = 275
no_of_epochs = 100
batch_size = 32
image_size = (224, 224)
train_dir = '/kaggle/input/gravity-spy-gravitational-waves/train/train'
val_dir = '/kaggle/input/gravity-spy-gravitational-waves/validation/validation'
test_dir = '/kaggle/input/gravity-spy-gravitational-waves/test/test'

def get_image_pixel(filepath):
    pixel = Image.open(filepath)
    pixel = pixel.convert('RGB')
    pixel = pixel.resize(image_size)
    pixel = np.array(pixel)
    pixel = pixel / 255.0
    return pixel

def load_images_from_directory(directory, exclude_classes=None):
    image_pixels = []
    image_labels = []
    columns = []
    
    for folderno, foldername in enumerate(os.listdir(directory)):
        if exclude_classes and foldername in exclude_classes:
            continue
        folderpath = os.path.join(directory, foldername)
        for filename in os.listdir(folderpath)[:datasize]:
            filepath = os.path.join(folderpath, filename)
            image_pixel = get_image_pixel(filepath)
            image_pixels.append(image_pixel)
            image_labels.append([folderno])
        columns.append(foldername)
        print(f'Loaded {foldername}: {len(image_pixels)} images')
    
    return np.array(image_pixels), pd.DataFrame(image_labels, columns=["label"]), columns

exclude_classes = ['None_of_the_Above', 'Paired_Doves', 'Air_Compressor', 'Wandering_Line']
X_train, y_train_df, columns_train = load_images_from_directory(train_dir, exclude_classes)
X_val, y_val_df, columns_val = load_images_from_directory(val_dir, exclude_classes)
X_test, y_test_df, columns_test = load_images_from_directory(test_dir, exclude_classes)

assert columns_train == columns_val == columns_test

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train = pd.DataFrame(oh_encoder.fit_transform(y_train_df), dtype='float64', columns=columns_train)
y_val = pd.DataFrame(oh_encoder.transform(y_val_df), dtype='float64', columns=columns_val)
y_test = pd.DataFrame(oh_encoder.transform(y_test_df), dtype='float64', columns=columns_test)

plt.imshow(X_train[0])
plt.show()

print(f'Training data shape: {X_train.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Validation data shape: {X_val.shape}')
print(f'Validation labels shape: {y_val.shape}')
print(f'Test data shape: {X_test.shape}')
print(f'Test labels shape: {y_test.shape}')

keras.backend.clear_session()

resnet = keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max')
resnet.trainable = False

resnet_model = Sequential([
    resnet,
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(len(columns_train), activation='softmax')
])

for layer in resnet_model.layers[1:]:
    layer.trainable = True

resnet_model.build(input_shape=(None, 224, 224, 3))

resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

resnet_model.summary()

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_df['label']), y=y_train_df['label'].values)
class_weights = dict(enumerate(class_weights))

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_resnet_model.keras', monitor='val_loss', save_best_only=True)

history = resnet_model.fit(
    X_train, y_train,
    epochs=no_of_epochs,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    callbacks=[early_stopping, checkpoint],
    class_weight=class_weights
)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.title('Training set')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Validation set')
plt.grid()
plt.ylim((0, 4))

resnet_model.save("resnet_model.keras")

test_loss, test_accuracy = resnet_model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

y_pred = resnet_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test.values, axis=1)

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=columns_train, yticklabels=columns_train)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
