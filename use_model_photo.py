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


def test_image(model_path, image_path, threshold=0.99):
    # Load the keypoint detection model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

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

    # Convert image back to RGB format
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # Display the image with keypoints and skeleton
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


# Call the function
if __name__ == "__main__":
    model_path = 'model_12.pth'  # Укажите путь к вашей модели
    image_path = '4.jpg'  # Укажите путь к вашему изображению
    test_image(model_path, image_path)
