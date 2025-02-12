import numpy as np
import torch
from torchvision.transforms import functional as tvF
from matplotlib import pyplot as plt
from src import utils

model = np.load("/Users/drewlinsley/Downloads/collected_maps/efficientvit_b0.r224_in1k/590_88868_176181_renders_00033.npy")
human = np.load("/Users/drewlinsley/Downloads/collected_maps/human/590_88868_176181_renders_00033.npy")

sim = utils.fast_normalized_emd(tvF.center_crop(torch.from_numpy(human), [model.shape[0], model.shape[1]]).numpy().mean(0), model)

print(sim)
