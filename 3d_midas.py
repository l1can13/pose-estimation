import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Определение соединений для скелета человека согласно COCO keypoints
skeleton = [
    # Руки
    (15, 13), (13, 11), (16, 14), (14, 12),
    # Туловище
    (11, 12), (5, 11), (6, 12),
    # Ноги
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Соединения головы
    (5, 0), (6, 0), (0, 1), (0, 2), (1, 3), (2, 4),
]

# Загрузка модели оценки глубины
model_type = "DPT_Large"  # Вы можете выбрать другие варианты: "DPT_Hybrid" или "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

# Загрузка трансформаций для модели
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.default_transform


def get_depth_map(image_path):
    """Возвращает карту глубины для изображения"""
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

    output = prediction.cpu().numpy()
    return output


def test_image(model_path, image_path, threshold=0.99):
    # Load the keypoint detection model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # Get the depth map
    depth_map = get_depth_map(image_path)

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Predict keypoints
    with torch.no_grad():
        predictions = model(image_tensor)[0]
        keypoints = predictions['keypoints'].detach().cpu().numpy()
        scores = predictions['scores'].detach().cpu().numpy()

    # Normalize depth values
    max_depth = np.max(depth_map)
    depth_values_normalized = depth_map / max_depth

    # Visualization in 3D
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(scores)):
        if scores[i] >= threshold:
            keypoint = keypoints[i]
            for start, end in skeleton:
                if keypoint[start][2] > threshold and keypoint[end][2] > threshold:
                    ax.plot(
                        [keypoint[start][0], keypoint[end][0]],
                        [keypoint[start][1], keypoint[end][1]],
                        [depth_values_normalized[int(keypoint[start][1]), int(keypoint[start][0])],
                         depth_values_normalized[int(keypoint[end][1]), int(keypoint[end][0])]],
                        "ro-", linewidth=2, markersize=5
                    )
            # Scale depth values for visualization
            ax.scatter(keypoint[:, 0], keypoint[:, 1], depth_values_normalized[[int(kp[1]) for kp in keypoint],
            [int(kp[0]) for kp in keypoint]],
                       c='b', s=35, edgecolors='w')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.view_init(elev=20., azim=45)  # Set the view angle
    plt.show()


# Call the function
if __name__ == "__main__":
    model_path = 'train_libs/model_12.pth'  # Укажите путь к вашей модели
    image_path = 'test.jpg'  # Укажите путь к вашему изображению
    test_image(model_path, image_path)
