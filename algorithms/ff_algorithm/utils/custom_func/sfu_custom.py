import torch
import matplotlib.pyplot as plt
from utils.common_func.lut_monitor import LUTErrorMonitor

_LUT_CACHE = {}
USE_INTERPOLATE = True # uv.sfu_interpolate

lut_error_monitor = LUTErrorMonitor()

def _ensure_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x) 
    return x

def _lookup_lut(x, func_name, compute_func, start, end, size):
    x = _ensure_tensor(x)
    device = x.device 
    dtype = x.dtype
    
    key = (func_name, size, start, end, str(device), str(dtype))
    
    if key not in _LUT_CACHE:
        grid = torch.linspace(start, end, steps=size, device=device, dtype=dtype)
        table = compute_func(grid)
        step = (end - start) / (size - 1)
        _LUT_CACHE[key] = (table, step, start)
    
    table, step, start_val = _LUT_CACHE[key]
    
    if table.device != device:
        table = table.to(device)
        
    idx_float = (x - start_val) / step # In HW, 1/step is pre-defined value
    idx_calc = idx_float.float()
    
    if not USE_INTERPOLATE:
        indices = torch.round(idx_calc).long()
        indices = torch.clamp(indices, 0, size - 1)
        return table[indices]
    
    else:
        idx_floor = torch.floor(idx_calc).long()
        idx_floor = torch.clamp(idx_floor, 0, size - 2)
        
        alpha = idx_float - idx_floor
        
        y0 = table[idx_floor]
        y1 = table[idx_floor + 1]
        
        slope = y1 - y0
        return y0 + slope * alpha


def lut_reciprocal(x, lut_ideal=True, lut_size=256):
    """
    H/W: Reciprocal LUT (1/x)
    Range: m in [1, 2]
    """
    x = _ensure_tensor(x)
    m, e = torch.frexp(x) # m in [0.5, 1)
    # Adjust m to be in [1, 2)
    m = m * 2.0
    e = e - 1
    
    # 1. Ideal
    res_ideal = None
    if lut_ideal or lut_error_monitor.enabled:
        out_ideal = 1.0 / m
        res_ideal = torch.ldexp(out_ideal, -e).to(dtype=x.dtype)
        
    # 2. LUT
    res_lut = None
    if not lut_ideal or lut_error_monitor.enabled:
        out_lut = _lookup_lut(m, 'reciprocal', lambda v: 1.0/v, 1.0, 2.0, lut_size)
        res_lut = torch.ldexp(out_lut, -e).to(dtype=x.dtype)
        
    # 3. Monitoring
    if lut_error_monitor.enabled and (res_ideal is not None) and (res_lut is not None):
        lut_error_monitor.update('reciprocal', res_ideal, res_lut)
        
    return res_ideal if lut_ideal else res_lut


def lut_sqrt(x, lut_ideal=True, lut_size=256):
    """
    H/W: Sqrt LUT
    Logic: x = m * 2^2k -> sqrt(x) = sqrt(m) * 2^k
    """
    x = _ensure_tensor(x)
    m, e = torch.frexp(x)
    # Adjust range: if e is odd, m needs doubling to make e even
    mask_odd = (e % 2 != 0)
    k = torch.div(e, 2, rounding_mode='floor')
    
    # If e was odd: exponent became 2k+1. We took k. The remaining +1 goes to mantissa.
    # x = m * 2^(2k+1) = (m*2) * 2^2k. So we sqrt(m*2).
    m_adjusted = m.clone()
    m_adjusted[mask_odd] *= 2.0
    
    res_ideal = None
    if lut_ideal or lut_error_monitor.enabled:
        out_ideal = torch.sqrt(m_adjusted)
        res_ideal = torch.ldexp(out_ideal, k).to(dtype=x.dtype)
        
    res_lut = None
    if not lut_ideal or lut_error_monitor.enabled:
        out_lut = _lookup_lut(m_adjusted, 'sqrt', torch.sqrt, 0.5, 2.0, lut_size)
        res_lut = torch.ldexp(out_lut, k).to(dtype=x.dtype)
        res_lut[x == 0] = 0
        
    if lut_error_monitor.enabled and (res_ideal is not None) and (res_lut is not None):
        lut_error_monitor.update('sqrt', res_ideal, res_lut)
    
    return res_ideal if lut_ideal else res_lut


def lut_exp(x, lut_ideal=True, lut_size=256):
    """
    Exp(x) = 2^k * exp(r)
    Range: r in [-ln2/2, ln2/2]
    """
    x = _ensure_tensor(x)
    ln2_val = 0.69314718
    ln2_inv_val = 1.44269504 # 1 / ln2 (precalculated in HW)
    ln2 = torch.tensor(ln2_val, device=x.device, dtype=x.dtype)
    ln2_inv = torch.tensor(ln2_inv_val, device=x.device, dtype=x.dtype)
    
    # k = torch.round(x / ln2)
    k = torch.round(x * ln2_inv)
    r = x - k * ln2
    
    res_ideal = None
    if lut_ideal or lut_error_monitor.enabled:
        out_ideal = torch.exp(r)
        res_ideal = torch.ldexp(out_ideal, k.int()).to(dtype=x.dtype)
        
    res_lut = None
    if not lut_ideal or lut_error_monitor.enabled:
        # LUT Range: [-ln2/2, ln2/2]
        bound = ln2_val / 2
        out_lut = _lookup_lut(r, 'exp', torch.exp, -bound, bound, lut_size)
        res_lut = torch.ldexp(out_lut, k.int()).to(dtype=x.dtype)
        
    if lut_error_monitor.enabled and (res_ideal is not None) and (res_lut is not None):
        lut_error_monitor.update('exp', res_ideal, res_lut)
        
    return res_ideal if lut_ideal else res_lut


def lut_log(x, lut_ideal=True, lut_size=256):
    """
    Log(x) = k*ln2 + log(m)
    Range: m in [1, 2]
    """
    x = _ensure_tensor(x)
    ln2_val = 0.69314718
    ln2 = torch.tensor(ln2_val, device=x.device, dtype=x.dtype)
    
    m, e = torch.frexp(x)
    m = m * 2
    e = e - 1
    
    res_ideal = None
    if lut_ideal or lut_error_monitor.enabled:
        out_ideal = torch.log(m)
        res_ideal = out_ideal + e * ln2
        
    res_lut = None
    if not lut_ideal or lut_error_monitor.enabled:
        out_lut = _lookup_lut(m, 'log', torch.log, 1.0, 2.0, lut_size)
        res_lut = out_lut + e * ln2
        
    if lut_error_monitor.enabled and (res_ideal is not None) and (res_lut is not None):
        lut_error_monitor.update('log', res_ideal, res_lut)
        
    return res_ideal if lut_ideal else res_lut


def lut_pow(base, exponent, lut_ideal=True, lut_size=256):
    """ H/W: Pow using Exp and Log (base^exp = e^(exp*ln(base))) """
    # x^y = exp(y * ln(x))
    # Note: This is efficient if base is constant (LUT_LOG called once)
    exponent = _ensure_tensor(exponent)
    base = _ensure_tensor(base)
    
    log_val = lut_log(base, lut_ideal=lut_ideal, lut_size=lut_size)
    
    return lut_exp(exponent * log_val, lut_ideal=lut_ideal, lut_size=lut_size)
    
################################################################
##############      SFU TEST FUNCTIONS    ######################
################################################################
def run_benchmark():
    # LUT Sizes
    lut_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    N = 10000
    DEVICE = "cuda"
    dtype = torch.float16
    # dtype = torch.float32
    errtype = 'abs' # 'ratio'
    print(f"Generating Inputs on {DEVICE}...")
    
    inputs_pos = torch.linspace(0.1, 10.0, N, device=DEVICE, dtype=dtype)
    inputs_full = torch.linspace(-3.0, 3.0, N, device=DEVICE, dtype=dtype)
    inputs_base = (torch.rand(N, device=DEVICE, dtype=dtype) * 5 + 0.1)
    inputs_exp = (torch.rand(N, device=DEVICE, dtype=dtype) * 2 - 1)

    results = {
        "Reciprocal": [], "Sqrt": [], "Exp": [], "Log": [], "Pow": []
    }

    print("Running Benchmark (Metric: Relative Error, Reference: FP32)...")

    for size in lut_sizes:
        # --- Reciprocal ---
        # 1. Ideal: FP32 (Golden Reference)
        ideal_32 = 1.0 / inputs_pos.float()
        # 2. LUT
        real_16 = lut_reciprocal(inputs_pos, lut_ideal=False, lut_size=size)
        # 3. Relative Error
        if errtype == 'ratio':
            err = torch.max(torch.abs((ideal_32 - real_16.float()) / (ideal_32 + 1e-8))).item()
        if errtype == 'abs':
            err = torch.max(torch.abs(ideal_32 - real_16.float())).item()
        results["Reciprocal"].append(err)

        # --- Sqrt ---
        ideal_32 = torch.sqrt(inputs_pos.float())
        real_16 = lut_sqrt(inputs_pos, lut_ideal=False, lut_size=size)
        if errtype == 'ratio':
            err = torch.max(torch.abs((ideal_32 - real_16.float()) / (ideal_32 + 1e-8))).item()
        if errtype == 'abs':
            err = torch.max(torch.abs(ideal_32 - real_16.float())).item()
        results["Sqrt"].append(err)

        # --- Exp ---
        ideal_32 = torch.exp(inputs_full.float())
        real_16 = lut_exp(inputs_full, lut_ideal=False, lut_size=size)
        if errtype == 'ratio':
            err = torch.max(torch.abs((ideal_32 - real_16.float()) / (ideal_32 + 1e-8))).item()
        if errtype == 'abs':
            err = torch.max(torch.abs(ideal_32 - real_16.float())).item()
        results["Exp"].append(err)

        # --- Log ---
        ideal_32 = torch.log(inputs_pos.float())
        real_16 = lut_log(inputs_pos, lut_ideal=False, lut_size=size)
        if errtype == 'ratio':
            err = torch.max(torch.abs((ideal_32 - real_16.float()) / (ideal_32 + 1e-8))).item()
        if errtype == 'abs':
            err = torch.max(torch.abs(ideal_32 - real_16.float())).item()
        results["Log"].append(err)

        # --- Pow ---
        # Pow = base^exp
        base_32 = inputs_base.float()
        exp_32 = inputs_exp.float()
        ideal_32 = torch.pow(base_32, exp_32)
        real_16 = lut_pow(inputs_base, inputs_exp, lut_ideal=False, lut_size=size)
        if errtype == 'ratio':
            err = torch.max(torch.abs((ideal_32 - real_16.float()) / (ideal_32 + 1e-8))).item()
        if errtype == 'abs':
            err = torch.max(torch.abs(ideal_32 - real_16.float())).item()
        results["Pow"].append(err)

    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v']
    for i, (func_name, errors) in enumerate(results.items()):
        plt.plot(lut_sizes, errors, marker=markers[i], label=func_name, linewidth=2)

    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Marginal Error (1e-3)
    plt.axhline(y=0.001, color='k', linestyle='--', alpha=0.3, label='FP16 Precision Limit (~0.1%)')

    plt.xlabel('LUT Size (Entries)', fontsize=12)
    # plt.ylabel('Max Relative Error (%)', fontsize=12)
    plt.ylabel('Max Absolute Error (vs Ideal)', fontsize=12)
    plt.title('Relative Accuracy vs LUT Size', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.xticks(lut_sizes, [str(s) for s in lut_sizes])
    
    plt.tight_layout()
    plt.savefig('LUT_diff.png')
    plt.show()

if __name__ == "__main__":
    run_benchmark()