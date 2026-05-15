import math
import torch
import time
from utils.orig_scff.get_pos_neg_imgs import add_outputs
# from utils.evaluate_scff import stdnorm
from utils.custom_func.optimizer_custom import adamW_step
from utils.common_func.plot_funcs import *
import user_variables as uv
from utils.custom_func.softplus_custom import lut_softplus_autograd
from utils.custom_func.sfu_custom import lut_error_monitor
from utils.custom_func.mean_custom import custom_mean
from utils.custom_func.pow_custom import custom_pow
           
# def cosine_lr(step, total_steps, lr_max, lr_min):
#     t = step / total_steps
#     return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))

def train(nets, device, optimizers, schedulers, threshold1, threshold2, dims_in, epochs, pool
            ,lamda, freezelayer, trainloader, alleps, lr, period, save_path):
    
    NL = len(nets)
    
    firstpass=True
    nbbatches = 0

    Dims = []
    best_loss = float("inf")
    loss_history = []
    pos_history = [[] for _ in range(NL)]
    neg_history = [[] for _ in range(NL)]
    
    start_time = time.time()
    
    # lut_error_monitor.enable()
    lut_error_monitor.disable()
    for epoch in range(epochs):
        print("Epoch", epoch)
        
        epoch_pos_sums = [0.0] * NL
        epoch_neg_sums = [0.0] * NL
        epoch_loss = 0.0
        nb_samples = 0
        
        for i, net in enumerate(nets):
            net.train()
        
        zeloader = trainloader

        goodness_pos = 0
        goodness_neg = 0
        
        lut_error_monitor.reset()
        
        for numbatch, (img, _) in enumerate(zeloader):
            nbbatches += 1
            img = img.to(device)

            for i in range(NL):
                
                if i == 0: # first image
                    img_norm = img
                else:
                    img_norm = x

                if nets[i].concat:
                    cv_out = nets[i](img_norm)
                    x, x_neg = add_outputs(cv_out)
                else:
                    x = nets[i](img_norm)
                    x_neg = nets[i](x_neg)
                
                # channel-wise accumulation (ch,y,x)->(y,x) 
                # yforgrad = nets[i].relu(x).pow(2).mean([1])
                # yforgrad_neg =nets[i].relu(x_neg).pow(2).mean([1])
                relu_pos = nets[i].relu(x)
                relu_neg =nets[i].relu(x_neg)
                relu_pow_pos = custom_pow(relu_pos, 2)
                relu_pow_neg = custom_pow(relu_neg, 2)
                yforgrad = custom_mean(relu_pow_pos, dim=1)
                yforgrad_neg = custom_mean(relu_pow_neg, dim=1)
                
                # yforgrad = (x * x.abs()).mean([1]) 
                # yforgrad_neg = (x_neg * x_neg.abs()).mean([1])

                # yforgrad = nets[i].act(x).pow(2).mean([1])
                # yforgrad_neg =nets[i].act(x_neg).pow(2).mean([1])
                
                epoch_pos_sums[i] += yforgrad.mean().item()
                epoch_neg_sums[i] += yforgrad_neg.mean().item()
                if i < freezelayer:
                    UNLAB = False
                else:
                    UNLAB = True

                if UNLAB and epoch<alleps[i]:
                    optimizers[i].zero_grad()
                    
                    penalty = lamda[i] * torch.norm(yforgrad, p=2, dim = (1,2)).mean()
                    sp_pos = lut_softplus_autograd(-yforgrad + threshold1[i], lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
                    sp_neg = lut_softplus_autograd(yforgrad_neg - threshold2[i], lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
                    sp_mean_pos = custom_mean(sp_pos, dim=[1,2]) # sp_pos.mean([1,2])
                    sp_mean_neg = custom_mean(sp_neg, dim=[1,2]) # sp_neg.mean([1,2])
                    loss_pos = custom_mean(sp_mean_pos) # sp_mean_pos.mean()
                    loss_neg = custom_mean(sp_mean_neg) # sp_mean_neg.mean()
                    loss = loss_pos + loss_neg + penalty
                    
                    loss.backward()
                    # optimizers[i].step()
                    adamW_step(optimizers[i], mode=uv.opm_type, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)

                    # 스케줄 적용 (new lr)
                    # lr_new = cosine_lr(numbatch, len(zeloader), lr_max=lr[i], lr_min=0.000002)
                    # for pg in optimizers[i].param_groups:
                    #     pg["lr"] = lr_new

                    nb_samples += x.size(0)
                    epoch_loss += loss.item() * x.size(0)
                    
                    # if (nbbatches+1)%period[i] == 0:
                    #     schedulers[i].step()
                    #     print(f'nbbatches {nbbatches+1} learning rate: {schedulers[i].get_last_lr()[0]}')
                
                x = pool[i](nets[i].act(x)).detach()
                x_neg = pool[i](nets[i].act(x_neg)).detach()

                if firstpass:
                    print("Layer", i, ": x.shape:", x.shape, "y.shape (after MaxP):", x.shape, end=" ")
                    _, channel, h, w = x.shape
                    Dims.append(channel * h * w)
                
            firstpass = False
            goodness_pos += (torch.mean(yforgrad.mean([1,2]))).item()
            goodness_neg += (torch.mean(yforgrad_neg.mean([1,2]))).item()

            if UNLAB and numbatch == len(zeloader) - 1:
                print(goodness_pos/len(zeloader), goodness_neg/len(zeloader))
        
        # Monotoring LUT errors (using only when monitor is enabled)
        lut_error_monitor.plot_errors(save_path=f"lut_error_accumulated_abs_{epoch}.png")
            
        for i in range(NL):
            pos_history[i].append(epoch_pos_sums[i] / len(zeloader))
            neg_history[i].append(epoch_neg_sums[i] / len(zeloader))
            print(f"Layer {i} - Pos: {pos_history[i][-1]:.4f}, Neg: {neg_history[i][-1]:.4f}")
            
        # save_weight_distribution(nets=nets, epoch=epoch)
        # for i in range(len(nets)):
        #     save_layer_heatmap(nets, layer_idx=i, epoch=epoch)
            
        if nb_samples > 0 and epoch > 0:
            avg_loss = epoch_loss / nb_samples
            loss_history.append(avg_loss) # 리스트에 추가
            print(f"avg_loss={avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save([net.state_dict() for net in nets], save_path)
                print(f"** Saved best model (loss={best_loss:.4f}) to {save_path}")
    
    # Save Dims for classifier (only for classifier in evaluate_scff.py)
    with open(save_path+".Dims.txt", "w") as f:
        for dim in Dims:
            f.write(f"{dim}\n")
        
    end_time = time.time()
    elapsed = end_time - start_time
    print("Training done.. (%.5f sec)" %(elapsed))
    
    plot_goodness_convergence(pos_history, neg_history, threshold1, threshold2)
    plot_loss(loss_history)

    return nets