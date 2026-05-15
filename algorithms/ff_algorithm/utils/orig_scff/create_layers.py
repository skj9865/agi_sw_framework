import torch
import torch.nn as nn
from utils.orig_scff.conv_funcs import Conv2d
import user_variables as uv

def create_layer(layer_config, device, act):
    net = Conv2d(layer_config["ch_in"], layer_config["channels"], (layer_config["kernel_size"], layer_config["kernel_size"]),
                  pad = layer_config["pad"], norm = "stdnorm", padding_mode = layer_config["padding_mode"], act = act)

    if layer_config["pooltype"] == 'Avg':
        pool = nn.AvgPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"], padding=layer_config["padding"], ceil_mode=True)
    else:
        pool = nn.MaxPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"], padding=layer_config["padding"], ceil_mode=True)
    
    extra_pool = nn.AvgPool2d(kernel_size= layer_config["extra_pool_size"], stride=layer_config["extra_pool_size"], padding=0, ceil_mode=True)
    net.to(device)
    if uv.dtype == torch.float16:
        net.half()

    return net, pool, extra_pool #, optimizer, scheduler


def build_networks(layer_model, concats, device, act):
    nets = []
    pools = []
    extra_pools = []

    for i, layer_config in enumerate(layer_model):
        net, pool, extra_pool = create_layer(layer_config, device=device, act=act[i])
        nets.append(net)
        pools.append(pool)
        extra_pools.append(extra_pool)

    for (net, concat) in zip(nets, concats):
        net.concat = concat

    return nets, pools, extra_pools