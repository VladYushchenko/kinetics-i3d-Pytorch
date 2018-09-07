import argparse
import torch
import os
import numpy as np
from scipy.stats import entropy
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from model.I3D_Pytorch import I3D
from loaders import UCF101
from evaluate_sample import _CHECKPOINT_PATHS, str2bool
from tqdm import tqdm


def inception_score(imgs, cuda=True, batch_size=32, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N >= batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    # model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    # model.eval()

    model = I3D(input_channel=3).cuda()
    model.eval()
    if args.imagenet_pretrained:
        state_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
        model.load_state_dict(state_dict)
        print('RGB checkpoint restored')
    else:
        state_dict = torch.load(_CHECKPOINT_PATHS['rgb'])
        model.load_state_dict(state_dict)
        print('RGB checkpoint restored')

    def get_pred(x):
        x = model.features_block(x)
        x = model.logits_block(x)
        x = x.squeeze(dim=2)
        x = x.squeeze(dim=2)
        x = x.squeeze(dim=2)
        # print(x.size())
        result = F.softmax(x, dim=1).data.cpu().numpy()
        return result

    # Get predictions
    # preds = np.zeros((N, 1000))
    preds = np.zeros((N, 400))

    for i, (batch, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.type(dtype)
        batch = Variable(batch).cuda()
        batch_size_i = batch.size()[0]

        result = get_pred(batch)
        preds[i*batch_size:i*batch_size + batch_size_i] = result

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", default='./data')
    parser.add_argument('--splits', type=int)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=224, help='Size of center crop')
    parser.add_argument('--imagenet_pretrained', type=str2bool, default='true')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    images = UCF101(root_path=args.data_root, spatial_transform=transform)

    if not args.splits:
        args.splits = len(os.listdir(args.data_root))
    if not args.batch_size:
        args.batch_size = len(images) // args.splits
    print('Batch size: {}\nSplits: {}'.format(args.batch_size, args.splits))
    print("Calculating Inception Score...")
    print(inception_score(images, batch_size=args.batch_size, splits=args.splits))
