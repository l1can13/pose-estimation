import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from dataset_helpers.coco_utils import get_coco, get_coco_kp

from torchvision_libs.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from train_libs.engine import train_one_epoch, evaluate

from torchvision_libs import utils, transforms as T

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f"Using device: {device}")

from torch.utils.data import Subset


def get_dataset(name, image_set, transform, subset_size=None):
    paths = {
        "coco": (r'D:\COCO Dataset\coco2017', get_coco, 91),
        "coco_kp": (r'D:\COCO Dataset\coco2017', get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)

    if subset_size is not None:
        indices = list(range(len(ds)))
        subset_indices = indices[:subset_size]
        ds = Subset(ds, subset_indices)

    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    dataset_name = 'coco_kp'
    model_name = 'keypointrcnn_resnet50_fpn'
    batch_size = 2
    epochs = 13
    workers = 4
    lr = 0.02
    momentum = 0.9
    weight_decay = 1e-4
    lr_step_size = 8
    lr_steps = [8, 11]
    lr_gamma = 0.1
    print_freq = 20
    output_dir = '..'
    resume_path = ''
    aspect_ratio_group_factor = 0
    test_only = False
    pretrained = True

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(dataset_name, "train_libs", get_transform(train=True), subset_size=500)
    dataset_test, _ = get_dataset(dataset_name, "val", get_transform(train=False), subset_size=50)

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.__dict__[model_name](num_classes=num_classes,
                                                              pretrained=pretrained)
    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
        lr_scheduler.step()
        if output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()},
                os.path.join(output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
