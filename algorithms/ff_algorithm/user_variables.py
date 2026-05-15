import torch

# eps = 1e-3
eps = 1e-10

# dtype = torch.float16
dtype = torch.float32

# opm = optimizer
# opm_type = 'lut' # type of optimizer: lut mode or normal mode
opm_type = 'normal'

# sfu = special function unit
# sfu_lut_ideal = False # Enable ideal LUT computations or not
sfu_lut_ideal = True # Enable ideal LUT computations or not
sfu_lut_size = 32 # The number of LUT entires in SFU module


# CUSTOM_CV = True
CUSTOM_CV = False



# NOT USING SO FAR
# sfu_interpolate = True # Applying interpolation in LUT or not