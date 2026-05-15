import torch

class CustomMeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, dim=None, keepdim=False):
        """
        Hardware View: Accumulator -> Constant Multiplier (1/N)
        """
        # N (Number of elements) 계산
        if dim is None:
            N = input_tensor.numel()
        elif isinstance(dim, int):
            N = input_tensor.size(dim)
        else:
            N = 1
            for d in dim:
                N *= input_tensor.size(d)
        
        # Backward에서 사용하기 위해 저장 (Context 저장)
        ctx.N = N
        ctx.input_shape = input_tensor.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        
        # 실제 연산 (Sum * 1/N)
        out = torch.mean(input_tensor, dim=dim, keepdim=keepdim)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Hardware View: Gradient Distributor * (1/N)
        Scalar(or smaller tensor) Gradient가 전체 Input 사이즈로 퍼짐
        """
        N = ctx.N
        input_shape = ctx.input_shape
        dim = ctx.dim
        keepdim = ctx.keepdim
        
        # grad_output을 input_shape로 복구(Expand)하기 위한 차원 처리
        if dim is not None and not keepdim:
            # 줄어든 차원을 다시 살려냄 (Unsqueeze)
            grad_expanded = grad_output
            
            # dim이 int일 경우 튜플로 변환
            dims = (dim,) if isinstance(dim, int) else dim
            
            # 앞에서부터 정렬하여 차례대로 unsqueeze (차원 복구)
            for d in sorted(dims):
                grad_expanded = grad_expanded.unsqueeze(d)
        else:
            grad_expanded = grad_output

        # Input과 동일한 크기로 확장 (Broadcasting)
        grad_input = grad_expanded.expand(input_shape)
        
        # 1/N 곱셈 (Mean의 미분은 1/N)
        return grad_input / N, None, None

# 래퍼 함수
def custom_mean(input_tensor, dim=None, keepdim=False):
    return CustomMeanFunction.apply(input_tensor, dim, keepdim)