import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft
from model import TransformerModel
from tqdm import tqdm


def imgs_to_patches(imgs, stride, img_res):
    filter_size_h, filter_size_w = img_res
    images_dataset = []

    for img in tqdm(imgs, desc="Cropping images"):
        image = plt.imread(img)
        # 3 dimensions (RGB) convert to YCrCb (take only Y -> luminance)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
            image = image[:, :, 0]
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        h, w = image.shape
        h_n = ((h - filter_size_h) // stride) + 1
        w_n = ((w - filter_size_w) // stride) + 1

        for i in range(h_n):
            for j in range(w_n):
                block = image[i * stride:(i * stride) + filter_size_h,
                              j * stride:(j * stride) + filter_size_w]
                images_dataset.append(block)
    
    images_dataset = np.array(images_dataset)
    images_dataset = np.clip(images_dataset, 0.0, 1.0, None)

    return images_dataset

class MyDataset(Dataset):
    def __init__(self, root_dir, img_res, transform=False):
        self.root_dir = root_dir
        self.transform = transform
        self.img_res = img_res
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        image_path = os.path.join(self.root_dir, image_index)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        H, W = self.img_res
        # img = img[0:H, 0:W]
        dim = (H, W)
        img = cv2.resize(img, dim)

        img = img.astype(np.float32) / 255
        img = torch.tensor(img)
        img = torch.unsqueeze(img, axis=2)
        return img

    
def psnr(img1, img2, PIXEL_MAX=1.0):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def merge(patches):
    assert type(patches) == torch.Tensor, "patches should be tensor"
    assert len(patches.shape) == 5, "patches should be [h_n, w_n, H, W, C]"
    h_n, w_n, H, W, C = patches.shape
    img = torch.zeros([h_n * H, w_n * W, C])
    for i in range(h_n):
        for j in range(w_n):
            img[i * H:(i + 1) * H, j * W:(j + 1) * W, :] = patches[i, j, :, :, :]
    return img


def propagation(u_in, feature_size, wavelength, z, dtype=torch.float32):
    # resolution of input field, should be: (num_images, num_channels, height, width, 2)
    field_resolution = u_in.size()

    # number of pixels
    num_y, num_x = field_resolution[2], field_resolution[3]

    # sampling inteval size
    dy, dx = feature_size

    # size of the field
    y, x = (dy * float(num_y), dx * float(num_x))

    # frequency coordinates sampling
    fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
    fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

    # momentum/reciprocal space
    FX, FY = np.meshgrid(fx, fy)

    # transfer function in numpy (omit distance)
    HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))

    # create tensor & upload to device (GPU)
    H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

    # reshape tensor and multiply
    H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))
    
    # multiply by distance
    H_exp = torch.mul(H_exp, z)

    # band-limited ASM - Matsushima et al. (2009)
    fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
    fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
    H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)
    
    # get real/img components
    H_real, H_imag = polar_to_rect(H_filter.to(u_in.device), H_exp)

    H = torch.stack((H_real, H_imag), 4)
    H = fft.ifftshift(H)
    H = torch.view_as_complex(H)

    U1 = torch.fft.fftn(fft.ifftshift(u_in), dim=(-2, -1), norm='ortho')
    U2 = H * U1
    u_out = fft.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))
    
    return u_out


def Projection(Uin, slm_size, wavelength, plane_distance, image):
    x = Uin
    Ax = propagation(x, slm_size, wavelength, plane_distance)
    real = Ax.real
    imag = Ax.imag
    mag, ang = rect_to_polar(real, imag)
    u = (torch.abs(mag) - image) * torch.exp(1j * ang)
    out = x - propagation(u, slm_size, wavelength, -plane_distance)
    return out


def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


class PhysicTransformer(torch.nn.Module):
    def __init__(self,
                 embed_dim=96,
                 num_heads=8,
                 attn_dropout=0.0,
                 relu_dropout=0.1,
                 res_dropout=0.1,
                 out_dropout=0.5,
                 layers=2,
                 slm_size=(8*1e-6, 8*1e-6),
                 wavelength=488*1e-9,
                 plane_distance=20*1e-2,
                 stage_num=2):
        super(PhysicTransformer, self).__init__()
        onelayer = []

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.stage_num = stage_num
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.layers = layers
        self.slm_size = slm_size
        self.wavelength = wavelength
        self.plane_distance = plane_distance

        for i in range(stage_num):
            onelayer.append(TransformerModel(embed_dim=self.embed_dim,
                                             num_heads=self.num_heads,
                                             attn_dropout=self.attn_dropout,
                                             relu_dropout=self.relu_dropout,
                                             res_dropout=self.res_dropout,
                                             out_dropout=self.out_dropout,
                                             layers=self.layers,
                                             ))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, image):
        layers_sym = []   # for computing symmetric loss
        x = x[..., 0] + 1j * x[..., 1]
        x = x.type(torch.complex64)
        
        for i in range(self.stage_num):
        
            x = Projection(x, self.slm_size, self.wavelength, self.plane_distance, image)
            x = self.fcs[i](x)
        
        x_final = propagation(x, self.slm_size, self.wavelength, self.plane_distance)
        x_final = torch.abs(x_final)

        return x_final

def ITV2d(f):
    ITV = []
    for i in range(f.shape[0]):
      img = f[i, :].reshape([1080, 1920])
      dh = diff2d_h(img)
      dv = diff2d_v(img)
      ITV.append(torch.sqrt(torch.sum(dh ** 2) + torch.sum(dv ** 2)))
    return torch.mean(torch.tensor(ITV, dtype=torch .float32))

def diff2d_v(X):
  X1 = X[0:-1, :]
  X2 = X[1:, :]
  return X2 - X1

def diff2d_h(X):
    X1 = X[:,0:-1]
    X2 = X[:,1:]
    return X2 - X1