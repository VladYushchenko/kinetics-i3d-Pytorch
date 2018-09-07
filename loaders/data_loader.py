# coding=utf-8
import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(root_dir):
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root_dir, class_to_idx):
    images = []
    root_dir = os.path.expanduser(root_dir)
    for target in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue
        # Note: Ordering is now preserved across all files. Evaluation + Training
        video_names = os.listdir(d)
        # video_names = [item for item in os.listdir(d) if is_image_file(os.path.join(d, item))]
        video_names.sort()
        path_images_all = video_names
        for video in video_names:
            if not os.path.isdir('{}/{}'.format(d, video)):
                continue
            frame_names = [item for item in os.listdir(video) if is_image_file(os.path.join(d, video, item))]
            frame_names.sort()
            path_images_all += frame_names

        for path in path_images_all:
            item = (path, class_to_idx[target])
            images.append(item)
        '''
        for root, _, fnames in sorted(os.walk(d)):
            # TODO: sorting function for root so ordered list of directories is read instead of random
            for joe_root, d_, imnames in sorted(os.walk(root)):
                #ensure reading of images is done in sequential order within the folder
                imnames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                for imname in imnames:
                    if is_image_file(imname):
                        path = os.path.join(joe_root, imname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        '''
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):

    try:
        import accimage
        return accimage.Image(path)
    except (IOError, ImportError):
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
        return pil_loader(path)


class UCF(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/breed1/xxx.png
        root/dog/breed2/xxy.png
        root/dog/breed3/xxz.png

        root/cat/breed1/123.png
        root/cat/breed2/nsdf3.png
        root/cat/breed3/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
