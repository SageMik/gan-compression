'''
This is the test model tailored for ResNet architecture.
'''
import ntpath
import os

import numpy as np
import torch
from skimage import color
from torch import nn, Tensor
from torchprofile import profile_macs
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple

from data.base_dataset import get_transform
from metric import get_fid
from models import networks
from models.base_model import BaseModel
from models.pix2pix_model import Pix2PixModel
from utils import util

import warnings

warnings.filterwarnings('ignore',
                        'Conversion from CIE-LAB, via XYZ to sRGB color space resulted in.*?',
                        category=UserWarning)


def rgb_to_lab(rgb) -> Tuple[Tensor, Tensor]:
    rgb = np.array(rgb)
    Lab = color.rgb2lab(rgb).astype(np.float32)
    Lab_t = transforms.ToTensor()(Lab)
    L = Lab_t[[0], ...] / 50.0 - 1.0
    ab = Lab_t[[1, 2], ...] / 110.0
    return L, ab


def lab_to_rgb(L, ab):
    """Convert an Lab tensor image to a RGB numpy output
    Parameters:
        L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
        ab (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

    Returns:
        rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
    """

    L2 = (L + 1.0) * 50.0  # torch.Size([32, 2, 256, 256])
    ab2 = ab * 110.0
    Labs = torch.cat([L2, ab2], dim=1).cpu().numpy()

    rgb = color.lab2rgb(Labs, channel_axis=1)
    rgb = (rgb - 0.5) * 2  # [0, 1] -> [-1, 1]
    return torch.tensor(rgb)


class ColorizationModel(Pix2PixModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='colorization')
        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        # reuse the pix2pix model
        Pix2PixModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A', 'real_B_rgb', 'fake_B_rgb']

    def compute_visuals(self):
        self.real_B_rgb = lab_to_rgb(self.real_A, self.real_B)
        self.fake_B_rgb = lab_to_rgb(self.real_A, self.fake_B)

    def evaluate_model(self, step):
        self.is_best = False

        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG.eval()

        fakes, names = [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
            self.set_input(data_i)
            self.test()
            fakes.append(self.fake_B_rgb.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10:
                    input_im = util.tensor2im(self.real_A[j])
                    real_im = util.tensor2im(self.real_B_rgb[j])
                    fake_im = util.tensor2im(self.fake_B_rgb[j])
                    util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png' % name), create_dir=True)
                    util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                    util.save_image(fake_im, os.path.join(save_dir, 'fake', '%s.png' % name), create_dir=True)
                cnt += 1

        fid = get_fid(fakes, self.inception_model, self.npz, device=self.device,
                      batch_size=self.opt.eval_batch_size, tqdm_position=2)
        if fid < self.best_fid:
            self.is_best = True
            self.best_fid = fid
        self.fids.append(fid)
        if len(self.fids) > 3:
            self.fids.pop(0)
        ret = {'metric/fid': fid, 'metric/fid-mean': sum(self.fids) / len(self.fids), 'metric/fid-best': self.best_fid}

        self.netG.train()
        return ret
