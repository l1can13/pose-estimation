import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model_path, device):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    return model


def process_video(video_path, output_path, model, device):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image_tensor)

        # Проверка наличия ключевых точек в предсказании
        if len(prediction[0]['keypoints']) > 0:
            keypoints = prediction[0]['keypoints'][0].cpu().numpy()[:, :2]
            for kp in keypoints:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
        else:
            print("Ключевые точки не обнаружены в этом кадре.")

        out.write(frame)

    cap.release()
    out.release()


# Использование функций
model = load_model('train_libs/model_12.pth', device)
process_video('test.mp4', 'result.mp4', model, device)
