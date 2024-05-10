import os
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from models.colorization_model import rgb_to_lab


class ColorizationDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        assert (opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')
        self.transform = get_transform(self.opt, toTensor=False, normalized=False)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        path = self.AB_paths[index]
        im = Image.open(path).convert('RGB')
        im = self.transform(im)
        L, ab = rgb_to_lab(im)
        return {'A': L,
                'B': ab,
                'A_paths': path,
                'B_paths': path}

    def __len__(self):
        return len(self.AB_paths)
