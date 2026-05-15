import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR
import json
import os

from utils.orig_scff.get_arguments import get_arguments_cifar, get_argument_mnist
from utils.orig_scff.get_loaders import get_train_cifar, get_train_mnist, get_train_svhn
from utils.orig_scff.create_layers import build_networks
from utils.classifier.evaluate_scff import evaluate_model
# from utils.classifier.evaluate_fast import evaluate_model
from utils.orig_scff.train import train
import user_variables as uv

class EvaluationConfig:
    def __init__(self, device, dims, dims_in, dims_out,stdnorm_out, out_dropout, Layer_out, pre_std, all_neurons):
        self.device = device
        self.dims = dims
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.stdnorm_out = stdnorm_out
        self.out_dropout = out_dropout
        self.Layer_out = Layer_out
        self.all_neurons = all_neurons
        self.pre_std = pre_std


def hypersearch(dims, dims_in, dims_out, Batchnorm, epochs
    ,all_neurons, NL, Layer_out, tr_and_eval
    ,pre_std, stdnorm_out, search, device_num, loaders,seed_num
    ,lr,weight_decay,gamma,threshold1,threshold2,lamda,out_dropout,act, concats, alleps, period, mode):

    trainloader, _, _, _ = loaders
    torch.manual_seed(seed_num)
    
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    optimizers = []; schedulers= []
    freezelayer = 0

    with open('config.json', 'r') as f:
        config = json.load(f)
        if mode == 'mnist':
            layer_model = config['MNIST_CNN_2']['layer_configs'][:NL]
        if mode == 'cifar':
            layer_model = config['CIFAR']['layer_configs'][:NL]
        if mode == 'svhn': # same network with CIFAR
            layer_model = config['CIFAR']['layer_configs'][:NL]
    
    nets, pools, _ = build_networks(layer_model, concats, device, act)
    
    for i, _ in enumerate(layer_model):
        optimizer = AdamW(nets[i].parameters(), lr=lr[i], weight_decay=weight_decay[i], eps=uv.eps)
        optimizers.append(optimizer)
        schedulers.append(ExponentialLR(optimizer, gamma[i]))
        
    config = EvaluationConfig(device=device, dims=dims, dims_in=dims_in, dims_out=dims_out, stdnorm_out = stdnorm_out, 
                              out_dropout=out_dropout, Layer_out=Layer_out,pre_std = pre_std, all_neurons = all_neurons)

    if mode == 'mnist':
        save_path = os.getcwd() + '/state/MNIST_best_state.pt'
    if mode == 'cifar':
        save_path = os.getcwd() + '/state/CIFAR10_best_state.pt'
    if mode == 'svhn':
        save_path = os.getcwd() + '/state/SVHN_best_state.pt'
    
    if tr_and_eval == 1: # if tr_and_eval == 1, both train and evaluate are processed
        _ = train(
            nets, device, optimizers,schedulers, threshold1, threshold2, dims_in, epochs, pools, lamda, freezelayer
            ,trainloader, alleps, lr, period, save_path)
    
    _, tacc = evaluate_model(layer_model, config, loaders, search, save_path, concats, device, act) 
        
    return tacc


def main(device_num,tr_and_eval 
         ,save_model, loaders, NL, lr, weight_decay
         , gamma, lamda, threshold1,threshold2, act, concats, alleps, seed_num, dims, period, mode):
    
    tacc = hypersearch(
        dims =  dims,
        dims_in = dims, 
        dims_out = dims,
        Batchnorm = False, 
        epochs = max(alleps), 
        all_neurons = False,
        NL = NL,
        Layer_out = [2,1,0],
        tr_and_eval = tr_and_eval,
        pre_std = True,
        stdnorm_out = True,
        search = False,
        device_num = device_num,
        loaders = loaders,
        seed_num = seed_num,
        lr = lr,
        weight_decay = weight_decay,
        gamma = gamma,
        threshold1 = threshold1,
        threshold2 = threshold2,
        lamda = lamda,
        out_dropout = 0.2,
        act = act,
        concats = concats,
        alleps = alleps,
        period=period,
        mode=mode
    )

    return tacc

if __name__ == "__main__":
    # mode = 'mnist'
    mode = 'cifar'
    # mode = 'svhn'

    # Parse arguments
    if mode == 'mnist':
        args = get_argument_mnist()
        loaders = get_train_mnist(batchsize=100)
    if mode == 'cifar':
        args = get_arguments_cifar()
        loaders = get_train_cifar(batchsize=100, augment="no")
    if mode == 'svhn':
        args = get_arguments_cifar() # same arguments with CIFAR
        loaders = get_train_svhn(batchsize=100)    
    # dims = (1,2,3)
    dims = (1, 2, 3)

    # Run training
    tsacc = main(
        device_num=args.device_num,
        tr_and_eval=args.tr_and_eval,
        save_model=args.save_model,
        loaders=loaders,
        NL=args.NL,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        lamda=args.lamda,
        threshold1=args.th1,
        threshold2=args.th2,
        act=args.act,
        concats=args.concats,
        alleps=args.alleps,
        seed_num=args.seed_num,
        dims= dims,
        period=args.period,
        mode=mode
    )

