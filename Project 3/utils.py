import gzip
import pickle
import numpy as np
import torch
from torchvision.transforms import v2


### File for small functions ###


# Read a zipped pickle file
def load_zipped_pickle(filename):
        with gzip.open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object
        

# Save as a zipped pickle file
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())

def test_onesplit(model, L, refield=32, sf=1):
    h, w = L.size()[-2:]

    top = slice(0, (h // 2 // refield + 1) * refield)
    bottom = slice(h - (h // 2 // refield + 1) * refield, h)
    left = slice(0, (w // 2 // refield + 1) * refield)
    right = slice(w - (w // 2 // refield + 1) * refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h // 2 * sf, :w // 2 * sf] = Es[0][..., :h // 2 * sf, :w // 2 * sf]
    E[..., :h // 2 * sf, w // 2 * sf:w * sf] = Es[1][..., :h // 2 * sf, (-w + w // 2) * sf:]
    E[..., h // 2 * sf:h * sf, :w // 2 * sf] = Es[2][..., (-h + h // 2) * sf:, :w // 2 * sf]
    E[..., h // 2 * sf:h * sf, w // 2 * sf:w * sf] = Es[3][..., (-h + h // 2) * sf:, (-w + w // 2) * sf:]
    return E

def data_transforms():
    transform = v2.RandomChoice([
        v2.RandomRotation(30),
        v2.RandomVerticalFlip(),
        v2.CenterCrop(256),
        v2.ColorJitter(brightness=0.4, contrast=0.4),
        v2.GaussianBlur(3, sigma=(0.1, 2.0)),
    ])
    return transform