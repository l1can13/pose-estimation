import datetime
import os
import time
import torch
from torch.utils.data import Subset
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn as my_custom_model
from torchvision.models.detection.mask_rcnn import MaskRCNN as MaskRCNN_custom
from torchvision_libs.group_by_aspect_ratio import GroupedBatchSampler as CustomGroupedBatchSampler, create_aspect_ratio_groups as create_aspect_ratio_groups_custom
from torchvision_libs import utils as my_utils, transforms as my_transforms
from torchvision_libs.coco_utils import get_coco as my_get_coco, get_coco_kp as my_get_coco_kp
from torchvision_libs.engine import train_one_epoch as my_train_one_epoch, evaluate as my_evaluate
from torchvision_libs.coco_eval import CocoEvaluator as MyCocoEvaluator
import math
import sys

device_custom = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device_custom}")


def get_coco_dataset_custom(image_set, transform, subset_size=None):
    dataset_path_custom = r'D:\COCO Dataset\coco2017'
    ds_fn_custom = my_get_coco if 'coco' in image_set else my_get_coco_kp
    num_classes_custom = 91 if 'coco' in image_set else 2

    dataset_custom = ds_fn_custom(dataset_path_custom, image_set=image_set, transforms=transform)

    if subset_size is not None:
        indices_custom = list(range(len(dataset_custom)))
        subset_indices_custom = indices_custom[:subset_size]
        dataset_custom = Subset(dataset_custom, subset_indices_custom)

    return dataset_custom, num_classes_custom


def get_transform_custom(train):
    transforms_custom = [my_transforms.ToTensor()]
    if train:
        transforms_custom.append(my_transforms.RandomHorizontalFlip(0.5))
    return my_transforms.Compose(transforms_custom)


def train_one_epoch_custom_custom(model_custom, optimizer_custom, data_loader_custom, device_custom, epoch_custom, print_freq_custom):
    model_custom.train()
    metric_logger_custom = my_utils.MetricLogger(delimiter="  ")
    metric_logger_custom.add_meter('lr', my_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header_custom = f'Epoch: [{epoch_custom}]'

    lr_scheduler_custom = None
    if epoch_custom == 0:
        warmup_factor_custom = 1. / 1000
        warmup_iters_custom = min(1000, len(data_loader_custom) - 1)
        lr_scheduler_custom = my_utils.warmup_lr_scheduler(optimizer_custom, warmup_iters_custom, warmup_factor_custom)

    for images_custom, targets_custom in metric_logger_custom.log_every(data_loader_custom, print_freq_custom, header_custom):
        images_custom = [image.to(device_custom) for image in images_custom]
        targets_custom = [{k: v.to(device_custom) for k, v in t.items()} for t in targets_custom]

        loss_dict_custom = model_custom(images_custom, targets_custom)
        losses_custom = sum(loss for loss in loss_dict_custom.values())

        loss_dict_reduced_custom = my_utils.reduce_dict(loss_dict_custom)
        losses_reduced_custom = sum(loss for loss in loss_dict_reduced_custom.values())

        loss_value_custom = losses_reduced_custom.item()

        if not math.isfinite(loss_value_custom):
            print(f"Loss is {loss_value_custom}, stopping training")
            print(loss_dict_reduced_custom)
            sys.exit(1)

        optimizer_custom.zero_grad()
        losses_custom.backward()
        optimizer_custom.step()

        if lr_scheduler_custom is not None:
            lr_scheduler_custom.step()

        metric_logger_custom.update(loss=losses_reduced_custom, **loss_dict_reduced_custom)
        metric_logger_custom.update(lr=optimizer_custom.param_groups[0]["lr"])

def main_custom():
    dataset_name_custom = 'coco_kp'
    model_name_custom = 'keypointrcnn_resnet50_fpn'
    batch_size_custom = 2
    epochs_custom = 13
    workers_custom = 4
    lr_custom = 0.02
    momentum_custom = 0.9
    weight_decay_custom = 1e-4
    lr_step_size_custom = 8
    lr_steps_custom = [8, 11]
    lr_gamma_custom = 0.1
    print_freq_custom = 20
    output_dir_custom = '..'
    resume_path_custom = ''
    aspect_ratio_group_factor_custom = 0
    test_only_custom = False
    pretrained_custom = True

    print("Loading data")
    train_transform_custom = get_transform_custom(train=True)
    test_transform_custom = get_transform_custom(train=False)

    dataset_custom, num_classes_custom = get_coco_dataset_custom(dataset_name_custom, train_transform_custom, subset_size=500)
    dataset_test_custom, _ = get_coco_dataset_custom(dataset_name_custom, test_transform_custom, subset_size=50)

    print("Creating data loaders")
    train_sampler_custom = torch.utils.data.RandomSampler(dataset_custom)
    test_sampler_custom = torch.utils.data.SequentialSampler(dataset_test_custom)

    if aspect_ratio_group_factor_custom >= 0:
        group_ids_custom = create_aspect_ratio_groups_custom(dataset_custom, k=aspect_ratio_group_factor_custom)
        train_batch_sampler_custom = CustomGroupedBatchSampler(train_sampler_custom, group_ids_custom, batch_size_custom)
    else:
        train_batch_sampler_custom = torch.utils.data.BatchSampler(train_sampler_custom, batch_size_custom, drop_last=True)

    data_loader_custom = torch.utils.data.DataLoader(dataset_custom, batch_sampler=train_batch_sampler_custom,
                                              num_workers=workers_custom, collate_fn=my_utils.collate_fn)

    data_loader_test_custom = torch.utils.data.DataLoader(dataset_test_custom, batch_size=1,
                                                   sampler=test_sampler_custom, num_workers=workers_custom,
                                                   collate_fn=my_utils.collate_fn)

    print("Creating model")
    model_custom = torchvision.models.detection.__dict__[model_name_custom](num_classes=num_classes_custom, pretrained=pretrained_custom)
    model_custom.to(device_custom)

    model_without_ddp_custom = model_custom
    params_custom = [p for p in model_custom.parameters() if p.requires_grad]
    optimizer_custom = torch.optim.SGD(params_custom, lr=lr_custom, momentum=momentum_custom, weight_decay=weight_decay_custom)

    lr_scheduler_custom = torch.optim.lr_scheduler.MultiStepLR(optimizer_custom, milestones=lr_steps_custom, gamma=lr_gamma_custom)

    if resume_path_custom:
        checkpoint_custom = torch.load(resume_path_custom, map_location=device_custom)
        model_without_ddp_custom.load_state_dict(checkpoint_custom['model'])
        optimizer_custom.load_state_dict(checkpoint_custom['optimizer'])
        lr_scheduler_custom.load_state_dict(checkpoint_custom['lr_scheduler'])

    if test_only_custom:
        my_evaluate(model_custom, data_loader_test_custom, device=device_custom)
        return

    print("Start training")
    start_time_custom = time.time()
    for epoch_custom in range(epochs_custom):
        my_train_one_epoch(model_custom, optimizer_custom, data_loader_custom, device_custom, epoch_custom, print_freq_custom)
        lr_scheduler_custom.step()
        if output_dir_custom:
            my_utils.save_on_master({'model': model_without_ddp_custom.state_dict(),
                                  'optimizer': optimizer_custom.state_dict(),
                                  'lr_scheduler': lr_scheduler_custom.state_dict()},
                                 os.path.join(output_dir_custom, f'model_{epoch_custom}.pth'))

        my_evaluate(model_custom, data_loader_test_custom, device=device_custom)

    total_time_custom = time.time() - start_time_custom
    total_time_str_custom = str(datetime.timedelta(seconds=int(total_time_custom)))
    print('Training time {}'.format(total_time_str_custom))


if __name__ == "__main__":
    main_custom()
