import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils.custom_func.conv_custom import Conv2dCustom
from utils.custom_func.sfu_custom import lut_sqrt, lut_reciprocal
from utils.custom_func.stdnorm_custom import stdnorm_lut
import user_variables as uv

    
class standardnorm(nn.Module):
    def __init__(self, dims = [1,2,3]):
        super(standardnorm, self).__init__()
        self.dims = dims
    
    def forward(self, x):
        
        # # x = x - torch.mean(x, dim=(self.dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(self.dims), keepdim=True))
        
        # m = torch.mean(x, dim=(self.dims), keepdim=True)
        # v = torch.sqrt(torch.mean(x**2, dim=(self.dims), keepdim=True) - m**2)
        # x = x - m
        # x = x / (uv.eps + v)
    
        # return x
        return stdnorm_lut(x, dims=self.dims, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
    
class L2norm(nn.Module):
    def __init__(self, dims = [1,2,3]):
        super(L2norm, self).__init__()
        self.dims = dims
    
    def forward(self, x):
        # return x / (x.norm(p=2, dim=(self.dims), keepdim=True) + uv.eps)
        # L2 Norm: sqrt( sum(x^2) )
        sum_sq = torch.sum(x * x, dim=self.dims, keepdim=True)
        norm_val = lut_sqrt(sum_sq, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
        denom = norm_val + uv.eps
        inv_denom = lut_reciprocal(denom, lut_ideal=self.lut_ideal, lut_size=self.lut_size)
        
        return x * inv_denom
    
class triangle(nn.Module):
    def __init__(self):
        super(triangle, self).__init__()

    def forward(self, x):
        x = x - torch.mean(x, axis=1, keepdims=True)
        return F.relu(x)

# with padding version
class Conv2d(nn.Module):
    
    def __init__(
        self, 
        input_channels, 
        output_channels, 
        kernel_size, 
        stride=1,         
        dilation=1,       
        pad=0, 
        batchnorm=False, 
        normdims=[1,2,3], 
        norm="stdnorm",
        bias=True, 
        dropout=0.0, 
        padding_mode="reflect", 
        concat=True, 
        act="relu"
    ):
        super(Conv2d, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.normdims = normdims
        self.concat = concat  # If True, input channels are split and processed separately because of concatenated pos/neg images
        self.relu = torch.nn.ReLU()
        
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        self.conv_layer = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            dilation=dilation,
            bias=bias,
            dtype=uv.dtype
        )
        
        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(self.conv_layer.weight)
        # Set padding parameters
        self.padding_mode = padding_mode
        self.F_padding = (pad, pad, pad, pad)  # Symmetric padding on all sides
        self.pad_val = pad

        # Define activation function
        if act == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = triangle()
        
        # Apply batch normalization if enabled
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.input_channels, affine=False)
        else:
            self.bn1 = nn.Identity()

        # Select normalization type (Standard Normalization or L2 Normalization)
        if norm == "L2norm":
            self.norm = L2norm(dims = normdims)
        elif norm == "stdnorm":
            self.norm = standardnorm(dims = normdims)
        else:
            self.norm = nn.Identity()

    def forward_standard(self, x):
        
        out = self.conv_layer(x)

        return out
    
    def forward_custom(self, x):
        
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        dH, dW = self.dilation
        sH, sW = self.stride

        # Output Size
        H_out = (H - dH * (kH - 1) - 1) // sH + 1
        W_out = (W - dW * (kW - 1) - 1) // sW + 1

        X_unf = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=0, stride=self.stride)
        weight_flat = self.conv_layer.weight.view(self.output_channels, -1)
        
        X_unf = X_unf.to(dtype=uv.dtype)
        weight_flat = weight_flat.to(dtype=uv.dtype)

        N, K, L = X_unf.shape
        Cout = self.output_channels
        
        # Accumulation Buffer
        Y16 = torch.zeros((N, Cout, L), device=x.device, dtype=uv.dtype)

        BLOCK_SIZE = 64 
        
        for k_start in range(0, K, BLOCK_SIZE):
            k_end = min(k_start + BLOCK_SIZE, K)
            
            # (N, BLOCK, L)
            val_x = X_unf[:, k_start:k_end, :] 
            
            # (Cout, BLOCK) -> (BLOCK, Cout)
            val_w = weight_flat[:, k_start:k_end].t() 
            
            # Partial Matrix Multiplication
            # (N, L, BLOCK) @ (BLOCK, Cout) -> (N, L, Cout)
            # transpose X to (N, L, BLOCK) to multiply
            prod = torch.matmul(val_x.transpose(1, 2), val_w)
            
            # Accumulate (transpose back to N, Cout, L)
            Y16 += prod.transpose(1, 2)

        out = Y16.view(N, Cout, H_out, W_out)

        if self.conv_layer.bias is not None:
            b = self.conv_layer.bias.view(1, Cout, 1, 1).to(dtype=uv.dtype)
            out = out + b

        return out

    def forward(self, x):
        x = self.bn1(x)
        if self.pad_val > 0:
            if self.padding_mode == 'reflect':
                x = F.pad(x, (self.pad_val, self.pad_val, self.pad_val, self.pad_val), mode='reflect')
                eff_pad = 0
            else:
                x = F.pad(x, (self.pad_val, self.pad_val, self.pad_val, self.pad_val), mode='constant', value=0)
                eff_pad = 0
        else:
            eff_pad = 0
        
        x = self.norm(x) # stardardization before convolutions
        
        if uv.CUSTOM_CV:
            return Conv2dCustom.apply(
                x, 
                self.conv_layer.weight, 
                self.conv_layer.bias, 
                self.stride, 
                (eff_pad, eff_pad), 
                self.dilation
            )
            
        else:
            # return self.forward_custom(x)
            return self.forward_standard(x)
        