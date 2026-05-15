import torch

class CustomPowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, exponent):
        """
        Hardware View: Multiplier Array (if exponent=2, x * x)
        """
        # Backward를 위해 input 저장
        ctx.save_for_backward(input_tensor)
        ctx.exponent = exponent
        
        return input_tensor.pow(exponent)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Hardware View: Multiplier 
        dL/dx = dL/dy * (exponent * x^(exponent-1))
        """
        input_tensor, = ctx.saved_tensors
        exponent = ctx.exponent
        
        # exponent가 2인 경우 (가장 흔한 케이스) -> 2 * x * grad
        if exponent == 2:
            grad_input = grad_output * input_tensor * 2.0
        else:
            grad_input = grad_output * exponent * input_tensor.pow(exponent - 1)
            
        return grad_input, None

# 래퍼 함수
def custom_pow(input_tensor, exponent=2.0):
    return CustomPowFunction.apply(input_tensor, exponent)