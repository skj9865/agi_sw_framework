
import argparse

def get_arguments_cifar():
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Contrastive Forward-Forward Training Script", add_help=False)

    parser.add_argument("--lr", nargs='+', type=float, default=[0.02, 0.001, 0.0004], help="Learning rate per layer")
    # parser.add_argument("--lr", nargs='+', type=float, default=[0.02, 0.02, 0.02], help="Learning rate per layer")
    parser.add_argument("--gamma", nargs='+', type=float, default=[0.99, 0.9, 0.99], help="LR decay rate per layer")
    parser.add_argument("--period", nargs='+', type=int, default=[500, 500, 500], help="LR decay rate period")
    parser.add_argument("--weight_decay", nargs='+', type=float, default=[0.0001, 0.0003, 0.0001], help="Weight decay")
    parser.add_argument("--lamda", nargs='+', type=float, default=[0.0008, 0.0004, 0.0016], help="Regularization lambda")
    
    # Threshold values for contrastive loss
    parser.add_argument("--th1", nargs='+', type=int, default=[1, 4, 5], help="Positive sample thresholds")
    parser.add_argument("--th2", nargs='+', type=int, default=[2, 5, 7], help="Negative sample thresholds")
    
    # parser.add_argument("--th1", nargs='+', type=int, default=[1, 5, 7], help="Positive sample thresholds")
    # parser.add_argument("--th2", nargs='+', type=int, default=[2, 6, 8], help="Negative sample thresholds")

    # Model structure and training
    parser.add_argument("--NL", type=int, default=3, help="Number of layers")
    parser.add_argument("--concats", type=tuple, default=(1, 0, 1), help="Concatenation setting per layer")
    parser.add_argument("--act", nargs='+', type=str, default=["triangle", "triangle", "relu"], help="Activation per layer")
    # parser.add_argument("--act", nargs='+', type=str, default=["relu", "relu", "relu"], help="Activation per layer")
    parser.add_argument("--alleps", nargs='+', type=int, default=[6, 6, 13], help="Epochs per layer")
    # parser.add_argument("--alleps", nargs='+', type=int, default=[15, 15, 15], help="Epochs per layer")
    # parser.add_argument("--alleps", nargs='+', type=int, default=[1, 1, 1], help="Epochs per layer")
    
    # Device settings
    parser.add_argument("--device_num", type=int, default=0, help="GPU device to use for training/testing")
    parser.add_argument("--seed_num", type=int, default=1234, help="Random seed for reproducibility")
    
    # Training options
    # parser.add_argument("--tr_and_eval", action="store_true", help="Enable training with evaluation")
    parser.add_argument("--tr_and_eval", type=int, default=1, help="Enable training with evaluation")
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    
    parser = argparse.ArgumentParser('ContrastFF script', parents=[parser])
    args = parser.parse_args()

    # Print argument values
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    return args


def get_argument_mnist():
    parser = argparse.ArgumentParser(description="SCFF MNIST Training")

    parser.add_argument("--lr", nargs='+', type=float, default=[0.004, 0.003, 0.0001], help="Learning rate per layer")
    parser.add_argument("--gamma", nargs='+', type=float, default=[0.7, 0.7, 0.7], help="LR decay rate per layer")
    parser.add_argument("--period", nargs='+', type=int, default=[500, 500, 500], help="LR decay rate period")
    parser.add_argument("--weight_decay", nargs='+', type=float, default=[0.0001, 0.0003, 0.0001], help="Weight decay")
    parser.add_argument("--lamda", nargs='+', type=float, default=[0.0008, 0.0004, 0.0016], help="Regularization lambda")
    
    # Threshold values for contrastive loss
    parser.add_argument("--th1", nargs='+', type=int, default=[1, 4, 5], help="Positive sample thresholds")
    parser.add_argument("--th2", nargs='+', type=int, default=[2, 5, 7], help="Negative sample thresholds")

    # Model structure and training
    parser.add_argument("--NL", type=int, default=3, help="Number of layers")
    parser.add_argument("--concats", type=tuple, default=(1, 1, 1), help="Concatenation setting per layer")
    parser.add_argument("--act", nargs='+', type=str, default=["relu", "relu", "relu"], help="Activation per layer")
    parser.add_argument("--alleps", nargs='+', type=int, default=[5, 1, 1], help="Epochs per layer")
    
    # Device settings
    parser.add_argument("--device_num", type=int, default=0, help="GPU device to use for training/testing")
    parser.add_argument("--seed_num", type=int, default=1234, help="Random seed for reproducibility")
    
    # Training options
    # parser.add_argument("--tr_and_eval", action="store_true", help="Enable training with evaluation")
    parser.add_argument("--tr_and_eval", type=int, default=1, help="Enable training with evaluation")
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    
    args = parser.parse_args()

    # Print argument values
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    return args