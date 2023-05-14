import cv2
import numpy as np
from keras.utils import img_to_array
from tensorflow.keras.models import load_model

# загружаем модель
model = load_model("my_model.h5")

# загружаем тестовое изображение
img = cv2.imread("test images/001.jpg")

# преобразуем изображение для подачи в модель
img = cv2.resize(img, (224, 224))
img_array = img_to_array(img)
img_array = img_array.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# делаем предсказание на модели
pred = model.predict(img_array)

# преобразуем предсказание в массив координат
coords = np.squeeze(pred)

# отображаем ключевые точки на изображении
for i in range(0, len(coords), 2):
    x, y = int(coords[i]), int(coords[i+1])
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

# отображаем изображение с ключевыми точками
cv2.imshow("Keypoints", img)
cv2.waitKey(0)
cv2.destroyAllWindows()