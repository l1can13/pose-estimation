import re
import pandas as pd
import matplotlib.pyplot as plt


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def parse_epoch_data(lines):
    epoch_data = {
        'epoch': [],
        'learning_rate': [],
        'loss_classifier': [],
        'loss_box_reg': [],
        'loss_keypoint': [],
        'loss_objectness': [],
        'loss_rpn_box_reg': []
    }
    pattern = re.compile(
        r'Epoch: \[(\d+)\].*lr: ([\d\.]+).*loss_classifier: ([\d\.]+).*loss_box_reg: ([\d\.]+).*'
        r'loss_keypoint: ([\d\.]+).*loss_objectness: ([\d\.]+).*loss_rpn_box_reg: ([\d\.]+)')
    for line in lines:
        match = pattern.search(line)
        if match:
            epoch_data['epoch'].append(int(match.group(1)))
            epoch_data['learning_rate'].append(float(match.group(2)))
            epoch_data['loss_classifier'].append(float(match.group(3)))
            epoch_data['loss_box_reg'].append(float(match.group(4)))
            epoch_data['loss_keypoint'].append(float(match.group(5)))
            epoch_data['loss_objectness'].append(float(match.group(6)))
            epoch_data['loss_rpn_box_reg'].append(float(match.group(7)))
    return pd.DataFrame(epoch_data)


def plot_metrics(df, title, y_label, metrics):
    plt.figure(figsize=(10, 5))
    for metric, label in metrics:
        plt.plot(df['epoch'], df[metric], marker='o', label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()


def parse_test_results(content):
    test_results = {
        'epoch': [],
        'bbox_AP': [],
        'bbox_AR': [],
        'keypoints_AP': [],
        'keypoints_AR': []
    }
    pattern_iou_bbox = re.compile(
        r"IoU metric: bbox\n"
        r".*?Average Precision  \(AP\) @\[.*?\| maxDets=100 \] = ([\d\.]+)\n"
        r".*?Average Recall     \(AR\) @\[.*?\| maxDets=100 \] = ([\d\.]+)",
        re.DOTALL
    )
    pattern_iou_keypoints = re.compile(
        r"IoU metric: keypoints\n"
        r".*?Average Precision  \(AP\) @\[.*?\| maxDets= 20 \] = ([\d\.]+)\n"
        r".*?Average Recall     \(AR\) @\[.*?\| maxDets= 20 \] = ([\d\.]+)",
        re.DOTALL
    )
    current_epoch = 0
    for epoch_block in content.split('Epoch: '):
        if epoch_block.strip():
            epoch_number = epoch_block.split()[0].strip('[]')
            if epoch_number.isdigit():
                current_epoch = int(epoch_number)

                # Extract all AP and AR values for bbox and keypoints
                bbox_aps = re.findall(r"Average Precision  \(AP\) @\[.*?\| maxDets=100 \] = ([\d\.]+)", epoch_block)
                bbox_ars = re.findall(r"Average Recall     \(AR\) @\[.*?\| maxDets=100 \] = ([\d\.]+)", epoch_block)
                keypoints_aps = re.findall(r"Average Precision  \(AP\) @\[.*?\| maxDets= 20 \] = ([\d\.]+)",
                                           epoch_block)
                keypoints_ars = re.findall(r"Average Recall     \(AR\) @\[.*?\| maxDets= 20 \] = ([\d\.]+)",
                                           epoch_block)

                if bbox_aps and bbox_ars:
                    test_results['epoch'].append(current_epoch)
                    test_results['bbox_AP'].append(sum(map(float, bbox_aps)) / len(bbox_aps))
                    test_results['bbox_AR'].append(sum(map(float, bbox_ars)) / len(bbox_ars))

                if keypoints_aps and keypoints_ars:
                    if current_epoch not in test_results['epoch']:
                        test_results['epoch'].append(current_epoch)
                    test_results['keypoints_AP'].append(sum(map(float, keypoints_aps)) / len(keypoints_aps))
                    test_results['keypoints_AR'].append(sum(map(float, keypoints_ars)) / len(keypoints_ars))

    return pd.DataFrame(test_results)


def main():
    file_path = 'resnet50.log'
    lines = read_file(file_path)
    df = parse_epoch_data(lines)
    mean_metrics_per_epoch = df.groupby('epoch').mean().reset_index()

    plot_metrics(mean_metrics_per_epoch, 'Mean Learning Rate per Epoch', 'Learning Rate',
                 [('learning_rate', 'Learning Rate')])

    # Plotting all loss metrics except keypoints loss
    plot_metrics(mean_metrics_per_epoch, 'Mean Loss Metrics per Epoch (Excluding Keypoint Loss)', 'Loss', [
        ('loss_classifier', 'Mean Classifier Loss'),
        ('loss_box_reg', 'Mean Box Reg Loss'),
        ('loss_objectness', 'Mean Objectness Loss'),
        ('loss_rpn_box_reg', 'Mean RPN Box Reg Loss'),
    ])

    # Plotting only keypoints loss
    plot_metrics(mean_metrics_per_epoch, 'Mean Keypoint Loss per Epoch', 'Loss', [
        ('loss_keypoint', 'Mean Keypoint Loss')
    ])

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    df_test_results = parse_test_results(content)

    plot_metrics(df_test_results, 'BBox IoU Metrics per Epoch', 'Metrics Value', [
        ('bbox_AP', 'BBox Average Precision (AP)'),
        ('bbox_AR', 'BBox Average Recall (AR)')
    ])

    plot_metrics(df_test_results, 'Keypoints IoU Metrics per Epoch', 'Metrics Value', [
        ('keypoints_AP', 'Keypoints Average Precision (AP)'),
        ('keypoints_AR', 'Keypoints Average Recall (AR)')
    ])


if __name__ == "__main__":
    main()
