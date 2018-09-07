import torch
from torch.autograd import Variable
import argparse
import numpy as np
import torch.nn.functional as F

from model.I3D_Pytorch import I3D

# _IMAGE_SIZE = 224
# _NUM_CLASSES = 400
#
# _SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': './data/pytorch_checkpoints/rgb_scratch.pkl',
    'flow': './data/pytorch_checkpoints/flow_scratch.pkl',
    'rgb_imagenet': './data/pytorch_checkpoints/rgb_imagenet.pkl',
    'flow_imagenet': './data/pytorch_checkpoints/flow_imagenet.pkl',
}

_LABEL_MAP_PATH = 'data/label_map.txt'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def launch_evaluation(args, test_data, test_classes):
    """Evaluate I3D network"""
    eval_type = args.eval_type
    if eval_type not in ['rgb', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
    rgb_logits, flow_logits = (0, 0)

    if eval_type in ['rgb', 'joint']:
        rgb_i3d = I3D(input_channel=3)
        rgb_i3d.eval()
        if args.imagenet_pretrained:
            state_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
            rgb_i3d.load_state_dict(state_dict)
            print('RGB checkpoint restored')
        else:
            state_dict = torch.load(_CHECKPOINT_PATHS['rgb'])
            rgb_i3d.load_state_dict(state_dict)
            print('RGB checkpoint restored')

        lol = np.load(_SAMPLE_PATHS['rgb'])
        rgb_sample = torch.from_numpy(lol)
        rgb_sample = Variable(rgb_sample.permute(0, 4, 1, 2, 3))
        print(np.max(lol), np.min(lol))

        print('RGB data loaded, shape=', str(test_data.data.size()))
        rbg_score, rgb_logits = rgb_i3d(test_data)

    if eval_type in ['flow', 'joint']:
        flow_i3d = I3D(input_channel=2)
        flow_i3d.eval()
        if args.imagenet_pretrained:
            state_dict = torch.load(_CHECKPOINT_PATHS['flow_imagenet'])
            flow_i3d.load_state_dict(state_dict)
            print('FLOW checkpoint restored')
        else:
            state_dict = torch.load(_CHECKPOINT_PATHS['flow'])
            flow_i3d.load_state_dict(state_dict)
            print('FLOW checkpoint restored')

        # flow_sample = torch.from_numpy(np.load(_SAMPLE_PATHS['flow']))
        # flow_sample = Variable(flow_sample.permute(0, 4, 1, 2 ,3))
        print('FLOW data loaded, shape=%s', str(test_data.data.size()))
        flow_score, flow_logits = flow_i3d(test_data)

    out_logits = (rgb_logits + flow_logits).squeeze(0)
    out_predictions = F.softmax(out_logits)
    sorted_indices = np.argsort(out_predictions.data.numpy())[::-1]

    out_logits = out_logits.data.numpy()
    out_predictions = out_predictions.data.numpy()

    print('Norm of logits: %f' % np.linalg.norm(out_logits))
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:
        print(out_predictions[index], out_logits[index], kinetics_classes[index])
