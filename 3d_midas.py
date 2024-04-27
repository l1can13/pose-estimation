import torch
import torchvision
import torchvision.transforms.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели оценки глубины
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to(device)
midas.eval()

# Трансформации для модели MiDaS
transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

skeleton = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 0), (6, 0), (0, 1), (0, 2), (1, 3), (2, 4),
]


def get_depth_map(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    return depth / np.max(depth)  # Нормализация глубины


def correct_keypoints(keypoints, depth_map, image_shape, threshold=0.1):
    # Получаем размеры изображения
    height, width = image_shape

    # Эвристика: коррекция на основе среднего расстояния между глазами
    for i in range(keypoints.shape[0]):
        eye_distance = np.linalg.norm(keypoints[i, 1, :2] - keypoints[i, 2, :2])
        head_width_estimate = eye_distance * 2

        for j in range(3, 5):  # Индексы ушей в COCO dataset
            ear_to_eye_distance = np.linalg.norm(keypoints[i, j, :2] - keypoints[i, 0, :2])

            if ear_to_eye_distance > head_width_estimate * threshold:
                # Сдвигаем ухо ближе к глазу
                keypoints[i, j, :2] = keypoints[i, 0, :2] + (keypoints[i, j, :2] - keypoints[i, 0, :2]) * (
                        head_width_estimate * threshold / ear_to_eye_distance)

                # Обновляем глубину уха, чтобы она была аналогична глубине глаз
                ear_depth = (depth_map[int(keypoints[i, 1, 1]), int(keypoints[i, 1, 0])] + depth_map[
                    int(keypoints[i, 2, 1]), int(keypoints[i, 2, 0])]) / 2
                keypoints[i, j, 2] = ear_depth

    return keypoints


def test_image(model_path, image_path, threshold=0.9, depth_threshold=0.7):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)[0]
        keypoints = predictions['keypoints'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        depth_map = get_depth_map(image_path)  # Получаем карту глубины

    # Коррекция ключевых точек
    keypoints = correct_keypoints(keypoints, depth_map, image.size, threshold)

    # Визуализация
    fig = plt.figure(figsize=(30, 10))
    ax2d = fig.add_subplot(131)
    axDepth = fig.add_subplot(132)
    ax3d = fig.add_subplot(133, projection='3d')

    # Отрисовка 2D скелета
    ax2d.imshow(image)
    ax2d.axis('off')

    # Отображение карты глубины
    axDepth.imshow(depth_map, cmap='gray')
    axDepth.axis('off')

    for i, score in enumerate(scores):
        if score >= threshold:
            keypoint = keypoints[i]
            for start, end in skeleton:
                if keypoint[start][2] > threshold and keypoint[end][2] > threshold:
                    ax2d.plot(
                        [keypoint[start][0], keypoint[end][0]],
                        [keypoint[start][1], keypoint[end][1]],
                        "ro-", linewidth=2, markersize=5
                    )

    # Отрисовка 3D скелета с использованием полного диапазона глубин
    min_depth = depth_map.min() * 10  # Минимальная глубина
    max_depth = depth_map.max() * 10  # Максимальная глубина

    for i, score in enumerate(scores):
        if score >= threshold:
            keypoint = keypoints[i]
            points_to_draw = []
            for start, end in skeleton:
                depth_start = depth_map[int(keypoint[start][1]), int(keypoint[start][0])] * 10
                depth_end = depth_map[int(keypoint[end][1]), int(keypoint[end][0])] * 10

                # Проверка разницы глубин между соединяемыми точками
                if np.abs(depth_start - depth_end) < depth_threshold:
                    ax3d.plot(
                        [keypoint[start][0], keypoint[end][0]],
                        [keypoint[start][1], keypoint[end][1]],
                        [depth_start, depth_end],
                        "ro-", linewidth=2, markersize=5
                    )
                    points_to_draw.append(keypoint[start])
                    points_to_draw.append(keypoint[end])

            # Отображаем активные точки
            if points_to_draw:
                points_to_draw = np.array(points_to_draw)
                ax3d.scatter(
                    points_to_draw[:, 0],
                    points_to_draw[:, 1],
                    depth_map[points_to_draw[:, 1].astype(int), points_to_draw[:, 0].astype(int)] * 10,
                    c='b', s=35, edgecolors='w'
                )

    # Установка пределов оси Z
    ax3d.set_xlim(0, image.width)
    ax3d.set_ylim(0, image.height)
    ax3d.set_zlim(min_depth, max_depth)  # Устанавливаем диапазон глубины

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Depth')
    ax3d.view_init(elev=20., azim=45)
    plt.show()


if __name__ == "__main__":
    test_image('train_libs/model_12.pth', '2.jpg')
