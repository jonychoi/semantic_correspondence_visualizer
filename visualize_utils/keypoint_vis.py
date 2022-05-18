import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def keypointer(args, model_name, index, confidence_map, save_dir, theme, upsample_mode = 'bilinear', name_plot = False, name = None):
    