import matplotlib.pyplot as plt
import torch
import torchvision
from train_libs.train import get_transform
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_image(model_path, image_path, transform, device, threshold=0.9):
    # Загрузите модель
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Переведите модель в режим оценки
    model.eval()

    # Преобразуйте ваше тестовое изображение
    image = Image.open(image_path).convert("RGB")
    image_tensor, _ = transform(image, {})  # передаем пустой словарь для target
    image_tensor = image_tensor.unsqueeze(0).to(device)  # добавьте размерность batch

    # Пропустите преобразованное изображение через модель
    with torch.no_grad():
        predictions = model(image_tensor)

    # Выведите изображение с предсказанными keypoints для всех обнаруженных объектов
    image = torchvision.transforms.ToPILImage()(image_tensor[0].cpu())
    plt.imshow(image)
    for person in predictions:  # Убрать [0] для итерации по всему списку детекций
        keypoints = person['keypoints'].cpu().numpy()
        # Возьмем только те детекции, для которых существует ключ 'scores'
        if 'scores' in person:
            keypoint_scores = person['scores'].cpu().numpy()
            # Отфильтровать точки согласно порогу вероятности
            for kp, score in zip(keypoints, keypoint_scores):
                if score > threshold:
                    plt.scatter(kp[:, 0], kp[:, 1], s=50, marker='.', c='r')
    plt.show()


# Протестируйте ваше изображение
if __name__ == "__main__":
    model_path = 'train_libs/model_12.pth'
    image_path = '1.jpg'
    transform = get_transform(train=False)
    test_image(model_path, image_path, transform, device, threshold=0.9)
