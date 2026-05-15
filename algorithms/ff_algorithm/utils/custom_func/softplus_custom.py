import torch
from utils.custom_func.sfu_custom import lut_exp, lut_log, lut_reciprocal

def softplus(z):
    return torch.log1p(torch.exp(z))

def softplus_clamp(z):
    z = torch.clamp(z, min=-11.0, max=11.0) # FP16 range
    return torch.log1p(torch.exp(z))        

def softplus_stable(z):
    return torch.maximum(z, torch.zeros_like(z)) + \
           torch.log1p(torch.exp(-torch.abs(z)))


def lut_softplus(z, lut_ideal=True, lut_size=256):
    """
    Softplus(z) = max(z, 0) + log(1 + exp(-|z|))
    Uses: lut_exp, lut_log
    """    
    # [Range] -|z| 는 (-inf, 0] 범위 -> exp 결과는 (0, 1] 범위
    neg_abs_z = -torch.abs(z)
    exp_val = lut_exp(neg_abs_z, lut_ideal=lut_ideal, lut_size=lut_size)
    
    # [Range] exp_val이 (0, 1] 이므로, (1 + exp_val)은 (1, 2] 범위가 됨
    # 이 범위는 lut_log의 입력 범위 [1, 2)와 완벽하게 매칭되어 정밀도가 높음
    log_input = 1.0 + exp_val
    log_val = lut_log(log_input, lut_ideal=lut_ideal, lut_size=lut_size)
    out = torch.maximum(z, torch.zeros_like(z)) + log_val
    
    return out

class LUTSoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, lut_ideal, lut_size):
        # Backward 계산을 위해 z 저장
        ctx.save_for_backward(z)
        ctx.lut_ideal = lut_ideal
        ctx.lut_size = lut_size
        
        # --- Forward Logic (위 함수와 동일) ---
        neg_abs_z = -torch.abs(z)
        exp_val = lut_exp(neg_abs_z, lut_ideal=lut_ideal, lut_size=lut_size)
        log_val = lut_log(1.0 + exp_val, lut_ideal=lut_ideal, lut_size=lut_size)
        out = torch.maximum(z, torch.zeros_like(z)) + log_val
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        lut_ideal = ctx.lut_ideal
        lut_size = ctx.lut_size
        
        # --- Backward Logic: Stable Sigmoid using LUT ---
        
        # [핵심] z가 양수든 음수든, exp 입력이 항상 음수가 되도록 -abs(z) 사용
        # 이렇게 하면 exp 결과가 절대 Overflow(Inf) 되지 않고 0~1 사이로 유지됨
        # exp_val = exp(-|z|)
        neg_abs_z = -torch.abs(z)
        exp_val = lut_exp(neg_abs_z, lut_ideal=lut_ideal, lut_size=lut_size)
        
        # denom = 1 + exp(-|z|)
        # 범위: [1.0, 2.0] -> lut_reciprocal이 가장 좋아하는 범위
        denom = 1.0 + exp_val
        
        # sig_temp = 1 / (1 + exp(-|z|))  == Sigmoid(|z|)
        sig_temp = lut_reciprocal(denom, lut_ideal=lut_ideal, lut_size=lut_size)
        
        # [Reconstruction]
        # z >= 0 일 때: Sigmoid(z) = Sigmoid(|z|) = sig_temp
        # z < 0  일 때: Sigmoid(z) = 1 - Sigmoid(-z) = 1 - sig_temp
        sigmoid_out = torch.where(z >= 0, sig_temp, 1.0 - sig_temp)
        
        grad_input = grad_output * sigmoid_out
        
        return grad_input, None, None
    

# 래퍼 함수
def lut_softplus_autograd(z, lut_ideal=True, lut_size=256):
    return LUTSoftplusFunction.apply(z, lut_ideal, lut_size)