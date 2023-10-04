import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Загрузка обученной модели
model = load_model('keypoints_model.h5')  # Укажите путь к вашей модели

# Загрузка тестового изображения
test_img_path = 'test_images/002.jpg'  # Укажите путь к вашему изображению
test_img = cv2.imread(test_img_path)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Изменение размера изображения на 224x224
test_img = cv2.resize(test_img, (224, 224))

# Подготовка изображения для предсказания моделью
test_img_array = tf.keras.preprocessing.image.img_to_array(test_img)
test_img_array = np.expand_dims(test_img_array, axis=0)
test_img_array = tf.keras.applications.resnet50.preprocess_input(test_img_array)

# Получение предсказаний модели
predictions = model.predict(test_img_array)

predicted_keypoints = decode_predictions(predictions)

# Нарисовать ключевые точки на изображении
for i in range(0, len(predicted_keypoints), 3):
    x, y, visibility = predicted_keypoints[i], predicted_keypoints[i + 1], predicted_keypoints[i + 2]
    if visibility > 0.5:  # Рисовать точку, если видимость больше 0.5 (пример)
        cv2.circle(test_img, (int(x), int(y)), 5, (255, 0, 0), -1)  # Красная точка радиусом 5

# Визуализация результата
plt.imshow(test_img)
plt.show()
