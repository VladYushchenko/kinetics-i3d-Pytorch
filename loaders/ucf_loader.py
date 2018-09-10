import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import functools
import json


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def video_loader(video_dir_path, frame_indices, image_ext, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:04d}.{}'.format(i + 1, image_ext))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            raise RuntimeError('Image {} not found!'.format(image_path))
    return video


def get_default_video_loader():
    return functools.partial(video_loader, image_loader=pil_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def find_classes(root_dir):
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root_path, n_video_samples, sample_duration):
    classes, classes_to_idx = find_classes(root_path)
    dataset = []
    for class_item in sorted(classes):

        d = os.path.join(root_path, class_item)
        if not os.path.isdir(d):
            continue

        # if i % 1000 == 0:
        #     print('dataset loading [{}/{}]'.format(i, len(video_names)))
        video_names = os.listdir(d)
        for name in video_names:
            video_path = os.path.join(d, name)
            if not os.path.exists(video_path):
                continue

            video_length = int(len([item for item in os.listdir(video_path) if is_image_file(item)]))
            if video_length <= 0:
                continue

            sample = {
                'video': video_path,
                'video_length': video_length,
                'label': classes_to_idx[class_item]
            }

            if video_length > sample_duration * n_video_samples:
                frames_needed = n_video_samples * (sample_duration - 1)
                possible_start_bound = video_length - frames_needed
                start = 0 if possible_start_bound == 0 else np.random.randint(0, possible_start_bound, 1)[0]
                subsequence_idx = np.linspace(start, start + frames_needed,
                                              sample_duration, endpoint=True, dtype=np.int32)
            elif video_length >= sample_duration:
                subsequence_idx = np.arange(0, sample_duration)
            else:
                raise RuntimeError("Length is too short id - {}, len - {}".format(video_path, video_length))
            sample['frame_indices'] = subsequence_idx
            dataset.append(sample)

    return dataset, classes


class UCF101(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 ext='jpg'):
        self.data, self.class_names = make_dataset(root_path, n_samples_for_each_video, sample_duration)

        self.ext = ext
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.ext)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip).permute(1, 0, 2, 3)

        target = self.data[index]['label']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
