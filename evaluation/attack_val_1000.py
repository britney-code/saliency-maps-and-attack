import glob
import os
from typing import List
import pandas as pd
import pretrainedmodels
import torch
import numpy as np
from PIL import Image
from imageio.v2 import imsave
import tqdm
import argparse
from Normalize import Normalize

opt = argparse.ArgumentParser()
opt.add_argument('--attack_name', type=str, default='pngp')
opt.add_argument('--input_dir', type=str, default='./dataset/val_rs')
opt.add_argument('--output_dir', type=str, default='./checkpoints/test/')
opt.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
opt.add_argument("--label_file", type=str, default='./dataset/val_rs.csv')
opt.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
opt.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
opt.add_argument("--mean", type=List[float], default=[0.5, 0.5, 0.5], help="mean.")
opt.add_argument("--std", type=List[float], default=[0.5, 0.5, 0.5], help="std.")
FLAGS = opt.parse_args()


def load_images(input_dir, batch_size):
    images = []
    filenames = []
    idx = 0
    for filepath in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir, filepath))
        image = image.resize((FLAGS.image_width, FLAGS.image_height)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images.append(np.array(image).astype(np.float32) / 255)
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)
            yield filenames, images
            filenames = []
            images = []
            idx = 0
    if idx > 0:
        images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)
        yield filenames, images


def get_labels(names, f2l):
    labels = []
    for name in names:
        labels.append(f2l[name] - 1)
    return torch.from_numpy(np.array(labels, dtype=np.int64))


def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    if output_dir is not None:
        check_or_create_dir(output_dir)
    if isinstance(images, torch.Tensor):
        images = np.transpose(images.detach().cpu().numpy(), (0, 2, 3, 1)) * 255
    for i, filename in enumerate(filenames):
        with open(os.path.join(output_dir, filename), 'wb') as f:
            imsave(f, images[i, :, :, :].astype('uint8'), format='png')


if __name__ == '__main__':
    total_batches = len(glob.glob(os.path.join(FLAGS.input_dir, '*'))) // FLAGS.batch_size
    model = torch.nn.Sequential(Normalize(FLAGS.mean, FLAGS.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())

    f2l = load_labels(os.path.join(FLAGS.label_file))
    for batch_idx, [filenames, images] in tqdm.tqdm(
            enumerate(load_images(os.path.join(FLAGS.input_dir), FLAGS.batch_size)),
            desc=f"Load images... attack {FLAGS.attack_name} ...", total=total_batches
    ):
        images = images.cuda()
        labels = get_labels(filenames, f2l).cuda()
        adv = graph(model, images, labels)
        save_images(adv, filenames, FLAGS.output_dir)
