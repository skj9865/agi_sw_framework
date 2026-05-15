import torch
import user_variables as uv
from utils.custom_func.sfu_custom import lut_sqrt, lut_reciprocal

def stdnorm (x, dims = [1,2,3]):

    # x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))

    # x = x - torch.mean(x, dim=(dims), keepdim=True)
    # x = x / (uv.eps + torch.std(x, dim=(dims), keepdim=True))

    m = torch.mean(x, dim=(dims), keepdim=True)
    v = torch.sqrt(torch.mean(x**2, dim=(dims), keepdim=True) - m**2)
    x = x - m
    x = x / (uv.eps + v)
    
    return x

def stdnorm_lut(x, dims=[1, 2, 3], lut_ideal=True, lut_size=256):
    """
    Standard Normalization using LUT Functions
    x = (x - mean) / sqrt(var + eps)
    """
    # 1. Mean (E[x])
    m = torch.mean(x, dim=dims, keepdim=True)
    
    # 2. Variance Calculation: E[x^2] - (E[x])^2
    mean_x2 = torch.mean(x * x, dim=dims, keepdim=True)
    var = mean_x2 - m * m
    
    var = torch.clamp(var, min=0.0)
    
    # 3. Standard Deviation (Std)
    v = lut_sqrt(var, lut_ideal=lut_ideal, lut_size=lut_size)
    
    # 4. Normalize
    # x = (x - m) / (v + eps)
    denom = v + uv.eps
    inv_denom = lut_reciprocal(denom, lut_ideal=lut_ideal, lut_size=lut_size)
    x_norm = (x - m) * inv_denom
    
    return x_norm