import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Загрузка данных из файла keypoints_old.csv
df = pd.read_csv('keypoints.csv')

df.fillna(-1, inplace=True)

# Определение путей к изображениям и ключевым точкам
image_paths = 'dataset/images/' + df['image'].values
keypoints = df.iloc[:, 1:].values

# Загрузка изображений и преобразование их в формат, подходящий для обучения модели
images = []
for path in image_paths:
    img = tf.keras.preprocessing.image.load_img(path, grayscale=True, target_size=(128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
    images.append(img)
images = tf.stack(images, axis=0) / 255.0

output_activation = 'sigmoid'

# Разделение данных на обучающую и проверочную выборки
train_images, val_images = images[:800], images[800:]
train_keypoints, val_keypoints = keypoints[:800], keypoints[800:]

# Определение архитектуры модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(36, activation=output_activation)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse')

# Обучение модели
history = model.fit(train_images, train_keypoints, epochs=50, batch_size=16, validation_data=(val_images, val_keypoints))

# Сохранение обученной модели
model.save('keypoints_model.h5')
