import torch

from torchvision_libs import utils
from train_libs.engine import evaluate
from train_libs.train import get_dataset, get_transform, device


def load_model(model_path):
    # Загружаем обученную модель
    model = torch.load(model_path)
    model.to(device)
    return model


def main():
    # Параметры
    dataset_name = 'coco_kp'  # Убедитесь, что используете правильное имя набора данных
    model_path = 'train_libs/model_12.pth'  # Укажите путь к обученной модели

    # Загружаем модель
    model = load_model(model_path)

    # Получаем тестовый набор данных
    _, _ = get_dataset(dataset_name, "train", get_transform(train=True))  # Для инициализации данных
    dataset_test, _ = get_dataset(dataset_name, "test2017", get_transform(train=False))

    # DataLoader для тестового набора
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn
    )

    # Выполняем оценку
    evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    main()
