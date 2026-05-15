import torch
from utils.custom_func.sfu_custom import lut_reciprocal, lut_sqrt, lut_pow

def adamW_step(optimizer, mode, lut_ideal, lut_size):
    """
    Replacing optimizers[i].step()
    """
    with torch.no_grad():
        for group in optimizer.param_groups:
            # Retrieve hypaer-parameters from Optimizer
            # These parameters are constant and unchanged during the training
            betas = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            # print(f"betas={betas[0]:.2f}, {betas[1]:2f} | lr={lr:2f} | wd={weight_decay:2f}")
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Retrieve Gradient
                g = p.grad
                
                # Optimizer Internal State (Momentum, ...)
                state = optimizer.state[p]

                # Initial State (First step)
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Variables
                state['step'] += 1
                step_i = state['step']
                exp_avg = state['exp_avg'] # moments
                exp_avg_sq = state['exp_avg_sq'] # velocity

                # ---------------- User Logic Implementation ----------------
                if mode == 'normal':
                    # Update moments
                    exp_avg.mul_(betas[0]).add_(g, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(g, g, value=1 - betas[1])
                    
                    # Bias correction
                    bias_c1 = 1 - betas[0] ** step_i
                    bias_c2 = 1 - betas[1] ** step_i

                    # Denom
                    # Prevent the case that bias_c2 becomes 0
                    denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)

                    # Step size
                    step_size = lr / bias_c1

                    # Parameter update
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    # Decoupled weight decay (AdamW Style)
                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)

                elif mode == 'lut':
                    exp_avg.mul_(betas[0]).add_(g, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(g, g, value=1 - betas[1])
                    
                    # Bias correction
                    # In HW, it can be accumulated each step
                    bias_c1 = 1 - betas[0] ** step_i
                    bias_c2 = 1 - betas[1] ** step_i
                    
                    h_v_sqrt = lut_sqrt(exp_avg_sq, lut_ideal, lut_size)
                    # h_v_sqrt = exp_avg_sq.sqrt()
                    h_bias_sqrt = lut_sqrt(bias_c2, lut_ideal, lut_size)
                    # h_bias_sqrt = (bias_c2 ** 0.5)
                    
                    h_bias_inv = lut_reciprocal(h_bias_sqrt, lut_ideal, lut_size)
                    denom = (h_v_sqrt * h_bias_inv).add_(eps)

                    # step_size = lr / bias_c1 -> lr * lut_recip(bias_c1)
                    bias_c1_inv = lut_reciprocal(bias_c1, lut_ideal, lut_size)
                    step_size = lr * bias_c1_inv

                    denom_inv = lut_reciprocal(denom, lut_ideal, lut_size)
                    p.addcmul_(exp_avg, denom_inv, value=-step_size)

                    # Decoupled weight decay
                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)
                else:
                    raise Exception("Argument error in AdamW Step")
                    

def sgd_step(optimizer):
    """
    Replacing optimizers[i].step()
    """
    with torch.no_grad():
        for group in optimizer.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad

                # 1. Weight Decay (L2 Penalty)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # 2. Momentum
                if momentum != 0:
                    param_state = optimizer.state[p]
                    
                    # Momentum Buffer 초기화
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # buf = momentum * buf + (1 - dampening) * d_p
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    # 3. Nesterov 여부에 따른 업데이트 방향(d_p) 결정
                    if nesterov:
                        # d_p = d_p + momentum * buf
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        # d_p = buf
                        d_p = buf

                # 4. Parameter Update
                p.add_(d_p, alpha=-lr)