import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import multiprocessing


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    # Загрузка COCO датасета
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CocoDetection(
        root='E:/COCO Dataset/coco2017/val2017',
        annFile='E:/COCO Dataset/coco2017/annotations/person_keypoints_val2017.json',
        transform=transform
    )

    # Создание DataLoader с использованием collate_fn
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Загрузка предварительно обученной модели keypointrcnn_resnet50_fpn
    model = keypointrcnn_resnet50_fpn(pretrained=True)

    # Замена последнего слоя классификации
    num_keypoints = 17  # количество ключевых точек человека
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor.kps_score_lowres = nn.ConvTranspose2d(in_features, num_keypoints, 2, 2, 0)
    model.roi_heads.keypoint_predictor.kps_score_lowres.bias.data.zero_()
    model.roi_heads.keypoint_predictor.kps_score_lowres.weight.data.zero_()

    # Обучение модели
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    criterion = nn.MSELoss()

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        for images, targets_list in train_loader:
            images = [image.to(device, dtype=torch.float32) for image in images]

            # Преобразование списка targets в словарь
            targets = []
            for target_list in targets_list:
                if len(target_list) != 0:
                    for target in target_list:
                        target_dict = {
                            'boxes': torch.empty((0, 4), device=device, dtype=torch.long),
                            'labels': torch.empty((0,), device=device, dtype=torch.long),
                            'keypoints': torch.tensor(target['keypoints'], device=device, dtype=torch.long),
                        }

                        print(target_dict)

                        targets.append(target_dict)

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Сохранение модели
    torch.save(model.state_dict(), 'keypoint_model.pth')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    multiprocessing.Process(target=train).start()
