import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

train_dir = os.path.join(os.path.dirname(__file__), '..', 'data_pp', 'train')
train_dir = os.path.abspath(train_dir)
print(f"Train dir: {train_dir}")

img_size = (224, 224)
batch_size = 32
epochs = 20  # Tùy chỉnh

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True
)
val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print("Classes:", class_names)

# EfficientNetB0 model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
base_model.trainable = False  # Freeze giai đoạn đầu

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train giai đoạn đầu
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    verbose=1
)

# Unfreeze toàn bộ model để fine-tune
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    verbose=1
)

# Lưu model và class_names
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sea_classifier.h5')
class_names_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_names.txt')
model.save(model_path)
with open(class_names_path, 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print("Training complete. Model and class names saved.")
