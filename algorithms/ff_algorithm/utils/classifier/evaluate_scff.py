
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
from utils.orig_scff.create_layers import build_networks
from utils.custom_func.optimizer_custom import adamW_step
from utils.custom_func.stdnorm_custom import stdnorm_lut
import user_variables as uv

class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]

def calculate_output_length(dims, nets, extra_pool, Layer, all_neurons):
    lengths = 0
    if all_neurons:
        for i, length in enumerate(dims):
            if i in Layer:
                lengths += length
    else:
        for i, length in enumerate(dims):
            #print(length)
            if i in Layer:
                len_after_pool = math.ceil((math.sqrt(length / nets[i].output_channels) - extra_pool[i].kernel_size) / extra_pool[i].stride + 1)
                lengths += len_after_pool*len_after_pool * nets[i].output_channels

    return lengths

def build_classifier(lengths, config):
    classifier = nn.Sequential(
        nn.Dropout(config.out_dropout),
        nn.Linear(lengths, 10)
    ).to(config.device)
    
    if uv.dtype == torch.float16:
        classifier.half() # half precision

    return classifier
    
def train_readout(classifier, nets, pool, extra_pool, loader, criterion, optimizer, config, epoch):
    # Training loop implementation
    classifier.train()
    correct = 0
    total = 0

    for i, (x, labels) in enumerate(loader):

        x = x.to(config.device)
        if uv.dtype == torch.float16:
            x = x.half()
        labels = labels.to(config.device)

        outputs = []
        
        with torch.no_grad():
            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm_lut(x, dims = config.dims_in, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
                    # x = torch.cat((x, x), dim=1)

                x = pool[j](net.act(net(x)))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm_lut(out, dims = config.dims_out, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

        outputs = torch.cat(outputs, dim = 1)    
        optimizer.zero_grad()
        outputs = classifier(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        adamW_step(optimizer, mode=uv.opm_type, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
        # optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

def test_readout(classifier, nets, pool, extra_pool, loader, criterion, config, epoch, mode):
    
    classifier.eval()
    running_loss = 0.
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (x, labels) in enumerate(loader):
       
            x = x.to(config.device)
            if uv.dtype == torch.float16:
                x = x.half()
            labels = labels.to(config.device)
            outputs = []
            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm_lut(x, dims = config.dims_in, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
                    # x = torch.cat((x, x), dim=1)
                    
                x = pool[j](net.act(net(x)))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm_lut(out, dims = config.dims_out, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

            outputs = torch.cat(outputs, dim = 1) 
            outputs = classifier(outputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    if mode == 'Val':
        print(f'Accuracy of the network on the 10000 '+ mode+ f' images: {100 * correct / total} %')
        print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

    return correct / total


def evaluate_model(layer_model, config, loaders, search, save_path, concats, device, act):
    print("Train & Test classifier (Released version)")
    n_epochs = 50
    nets, pool, extra_pool = build_networks(layer_model, concats, device, act)
    
    saved_states = torch.load(save_path)
    for net, state in zip(nets, saved_states):
        net.load_state_dict(state)
    
    Dims = []
    with open(save_path+".Dims.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            Dims.append(int(line.strip()))
        
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    lengths = calculate_output_length(Dims, nets, extra_pool, config.Layer_out, config.all_neurons)
    print(lengths)
    classifier = build_classifier(lengths, config)
    
    _, valloader, testloader, suptrloader = loaders
    # Optimizer and criterion setup
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, eps=uv.eps)
    # optimizer = optim.SGD(classifier.parameters(), lr=0.001)
    # optimizer = optim.SGD(classifier.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD(classifier.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4, nesterov=True)
    lr_scheduler = CustomStepLR(optimizer, nb_epochs=n_epochs)
    criterion = nn.CrossEntropyLoss()

    if not search:
        valloader = testloader
    # Main evaluation loop
    for _, net in enumerate(nets):
        net.eval()

    for epoch in range(n_epochs):
        acc_train = train_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, optimizer, config, epoch)
        lr_scheduler.step()
        if epoch % 20 == 0 or epoch == 49:
            print(f'Accuracy of the network on the 50000 train images: {100 * acc_train} %')
            acc_train = test_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, config,epoch, 'Train')
            acc_val = test_readout(classifier, nets, pool, extra_pool, valloader, criterion, config,epoch, 'Val')

    torch.set_rng_state(current_rng_state)
    
    return acc_train, acc_val
