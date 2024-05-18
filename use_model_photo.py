import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F

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


def load_model(model_path, backbone_name='resnet50'):
    # Загрузка и настройка backbone
    backbone = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=False)
    model = KeypointRCNN(backbone, num_classes=2, pretrained=False)  # num_classes для фона и человека
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


def test_image(model_path, image_path, backbone_name='resnet50', threshold=0.95):
    # Load the keypoint detection model
    model = load_model(model_path, backbone_name)

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Predict keypoints
    with torch.no_grad():
        predictions = model(image_tensor)[0]
        keypoints = predictions['keypoints'].detach().cpu().numpy()
        scores = predictions['scores'].detach().cpu().numpy()

    # Convert PIL image to OpenCV format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Plot keypoints and skeleton
    for i in range(len(scores)):
        if scores[i] >= threshold:
            keypoint = keypoints[i]
            for kp in keypoint:
                if kp[2] > threshold:
                    cv2.circle(image_cv2, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)
            for start, end in skeleton:
                start_point = tuple(keypoint[start, :2].astype(int))
                end_point = tuple(keypoint[end, :2].astype(int))
                if keypoint[start, 2] > threshold and keypoint[end, 2] > threshold:
                    cv2.line(image_cv2, start_point, end_point, (0, 255, 0), 2)

    # Convert image back to RGB format and display
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    model_path = 'resnet101.pth'  # Путь к вашей модели
    image_path = '1.jpg'  # Путь к вашему изображению
    backbone_name = 'resnet101'  # Указывается нужный backbone
    test_image(model_path, image_path, backbone_name)
