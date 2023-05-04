import argparse
import os
import numpy as np

from torchvision.utils import save_image

from torch.autograd import Variable

from torch import nn
import torch

os.makedirs("test_images_512_bale (2000e)", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=12, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt, unknown = parser.parse_known_args()
print(opt)

cuda = torch.cuda.is_available()
print(torch.cuda.is_available())

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Initialize generator and discriminator
generator = Generator()

if cuda:
    generator.cuda()

# Initialize weights
generator.load_state_dict(torch.load('/homes/zcoster/CIS732_Project/models/generator_2000e_512_bale.pth'))
generator.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for image_num in range(0, 5):
    # -----------------
    #  Train Generator
    # -----------------

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (10, opt.latent_dim))))

    # Generate a batch of images
    gen_imgs = generator(z)

    save_image(gen_imgs.data[:25], "test_images_512_bale (2000e)/%d.png" % image_num, nrow=5, normalize=True)
    print('Image {} saved.'.format(image_num))
