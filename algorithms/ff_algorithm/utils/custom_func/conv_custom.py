import torch
import torch.nn.functional as F
import user_variables as uv

FW_PARALLEL = 16 # 16
BW_PARALLEL = 1 # 64

class Conv2dCustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation):
        
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        
        input = input.to(dtype=uv.dtype)
        weight = weight.to(dtype=uv.dtype)
        if bias is not None:
            bias = bias.to(dtype=uv.dtype)

        N, C, H, W = input.shape
        out_ch, in_ch, kH, kW = weight.shape
        
        # Output Size
        H_out = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
        
        # 1. Unfold (im2col) -> (N, K, L)
        input_unf = F.unfold(input, kernel_size=(kH, kW), dilation=dilation, padding=padding, stride=stride)
        N, K, L = input_unf.shape
        
        # 2. Weight Flatten -> (Out, K)
        weight_flat = weight.view(out_ch, -1)
        
        # =================================================================
        # [Forward Optimization] Block 단위 Matrix Multiplication
        # 식: Output(N, L, Out) = Input(N, L, K) @ Weight.T(K, Out)
        # =================================================================
        BLOCK_SIZE = FW_PARALLEL
        
        # Accumulation Buffer
        accm = torch.zeros((N, L, out_ch), device=input.device, dtype=uv.dtype)
        
        # Transpose Input for efficient matmul: (N, K, L) -> (N, L, K)
        input_unf_T = input_unf.transpose(1, 2)
        weight_flat_T = weight_flat.t() # (K, Out)

        for k_start in range(0, K, BLOCK_SIZE):
            k_end = min(k_start + BLOCK_SIZE, K)
            
            # Block Slice
            # val_x: (N, L, Block)
            val_x = input_unf_T[:, :, k_start:k_end]
            # val_w: (Block, Out)
            val_w = weight_flat_T[k_start:k_end, :]
            
            # MatMul & Accumulate
            accm += torch.matmul(val_x, val_w)

        # Reshape: (N, L, Out) -> (N, Out, L) -> (N, Out, H_out, W_out)
        output = accm.transpose(1, 2).view(N, out_ch, H_out, W_out)

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        ctx.save_for_backward(input, weight, bias)
        ctx.input_shape = input.shape
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward Pass: FP16 + Block Accumulation (FP32 Buffer)
        """
        # breakpoint()
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        input_shape = ctx.input_shape
        
        N, C, H, W = input_shape
        out_ch, in_ch, kH, kW = weight.shape
        
        grad_output = grad_output.to(dtype=uv.dtype)
        grad_output_flat = grad_output.view(N, out_ch, -1) # (N, Out, L)
        
        grad_input = grad_weight = grad_bias = None

        # -------------------------------------------------
        # 1. Gradient w.r.t Bias
        # -------------------------------------------------
        if ctx.needs_input_grad[2]: 
            grad_bias = grad_output.sum(dim=(0, 2, 3)).to(dtype=uv.dtype)

        # -------------------------------------------------
        # 2. Gradient w.r.t Weight (dW)
        # 식: dW(Out, K) = Grad(Out, N*L) @ Input.T(N*L, K)
        # 공통 차원: N*L (Spatial * Batch) -> 매우 큼 -> Block 필수
        # -------------------------------------------------
        if ctx.needs_input_grad[1]:
            input_unf = F.unfold(input, kernel_size=(kH, kW), dilation=dilation, padding=padding, stride=stride)
            
            # Reshape for MatMul
            # Grad: (N, Out, L) -> (Out, N*L)
            grad_mat = grad_output_flat.permute(1, 0, 2).reshape(out_ch, -1)
            # Input: (N, K, L) -> (K, N*L) -> (N*L, K) (Transposed for multiplication)
            input_mat_T = input_unf.permute(1, 0, 2).reshape(in_ch * kH * kW, -1).t() 
            
            M = input_mat_T.shape[0] # N * L (Common Dimension)
            
            grad_weight_accm = torch.zeros((out_ch, in_ch * kH * kW), device=input.device, dtype=uv.dtype)
            
            SPATIAL_BLOCK_SIZE = BW_PARALLEL
            
            for m_start in range(0, M, SPATIAL_BLOCK_SIZE):
                m_end = min(m_start + SPATIAL_BLOCK_SIZE, M)
                
                # val_g: (Out, Block)
                val_g = grad_mat[:, m_start:m_end]
                # val_i: (Block, K)
                val_i = input_mat_T[m_start:m_end, :]
                
                # MatMul (FP32 casting for safety during mul)
                # (Out, Block) @ (Block, K) -> (Out, K)
                # prod = torch.matmul(val_g.float(), val_i.float())
                prod = torch.matmul(val_g, val_i)
                grad_weight_accm += prod
            
            grad_weight = grad_weight_accm.view_as(weight).to(dtype=uv.dtype)

        # -------------------------------------------------
        # 3. Gradient w.r.t Input (dX)
        # 식: dX(K, N*L) = Weight.T(K, Out) @ Grad(Out, N*L)
        # 공통 차원: Out (Output Channels) -> Block 적용
        # -------------------------------------------------
        if ctx.needs_input_grad[0]: 
            weight_flat = weight.view(out_ch, -1) # (Out, K)
            
            # Grad: (Out, N*L)
            grad_mat = grad_output_flat.permute(1, 0, 2).reshape(out_ch, -1)
            
            K_dim = in_ch * kH * kW
            M_dim = grad_mat.shape[1] # N * L
            
            grad_input_accm = torch.zeros((K_dim, M_dim), device=input.device, dtype=uv.dtype)
            
            CHANNEL_BLOCK_SIZE = BW_PARALLEL # 64
            
            for c_start in range(0, out_ch, CHANNEL_BLOCK_SIZE):
                c_end = min(c_start + CHANNEL_BLOCK_SIZE, out_ch)
                
                # val_w: (K, Block) - Weight Transpose
                val_w = weight_flat[c_start:c_end, :].t()
                # val_g: (Block, N*L)
                val_g = grad_mat[c_start:c_end, :]
                
                # MatMul
                prod = torch.matmul(val_w, val_g)
                grad_input_accm += prod
                
            # Reshape & Fold
            # (K, N*L) -> (K, N, L) -> (N, K, L)
            grad_input_unf = grad_input_accm.view(K_dim, N, -1).permute(1, 0, 2).to(dtype=uv.dtype)
            
            grad_input = F.fold(grad_input_unf, output_size=(H, W), kernel_size=(kH, kW), 
                                dilation=dilation, padding=padding, stride=stride)

        return grad_input, grad_weight, grad_bias, None, None, None, None