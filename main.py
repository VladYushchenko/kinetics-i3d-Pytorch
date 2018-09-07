import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from loaders import UCF, UCF101
from evaluate_sample import str2bool, launch_evaluation


def load_dataset(args):
    transform = transforms.Compose([
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    print('Started loading train...')
    train_path = os.path.join(args.dataset_path, 'train')
    training_data = UCF101(train_path, spatial_transform=transform)
    print(len(training_data))

    print('Started loading test...')
    test_path = os.path.join(args.dataset_path, 'test')
    test_data = UCF101(test_path, spatial_transform=transform)
    print(len(test_data))

    train_loader = DataLoader(training_data, batch_size=args.batch_size, drop_last=True, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True, num_workers=2, shuffle=True)

    return train_loader, training_data.class_names, test_loader, test_data.class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser('I3D Network with UCF101 dataset')
    parser.add_argument('--dataset_path', default='./data', help='Path to model state_dict')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--frame_number', type=int, default=16,
                        help='Number of video_frames to use (should be a multiple of 8)')
    parser.add_argument('--image_size', type=int, default=224, help='Size of center crop')
    parser.add_argument('--eval_type', type=str, default='rgb',
                        help='rgb, flow, or joint')
    parser.add_argument('--imagenet_pretrained', type=str2bool, default='true')
    args = parser.parse_args()
    train_loader, train_classes, test_loader, test_classes = load_dataset(args)

    for batch_id, (test_data, test_labels) in enumerate(test_loader):
        print('\nCurrent label {}'.format(test_classes[test_labels]))
        launch_evaluation(args, test_data, test_classes)
