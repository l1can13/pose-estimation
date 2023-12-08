import matplotlib.pyplot as plt
import torch
import torchvision
from train_libs.train import get_transform
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_image(model_path, image_path, transform, device):
    # 1. Загрузите модель
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # 2. Переведите модель в режим оценки
    model.eval()

    # 3. Преобразуйте ваше тестовое изображение
    image = Image.open(image_path).convert("RGB")
    image_tensor, _ = transform(image, {})  # передаем пустой словарь для target
    image_tensor = image_tensor.unsqueeze(0).to(device)  # добавьте размерность batch

    # 4. Пропустите преобразованное изображение через модель
    with torch.no_grad():
        prediction = model(image_tensor)

    # Выведите изображение с предсказанными keypoints
    image = torchvision.transforms.ToPILImage()(image_tensor[0].cpu())
    plt.imshow(image)
    keypoints = prediction[0]['keypoints'][0].cpu().numpy()
    keypoints = keypoints[:, :2]  # выберите x, y координаты (без вероятности)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, marker='.', c='r')
    plt.show()


# Протестируйте ваше изображение
if __name__ == "__main__":
    model_path = 'train_libs/model_12.pth'
    image_path = '3.jpg'
    transform = get_transform(train=False)
    test_image(model_path, image_path, transform, device)
