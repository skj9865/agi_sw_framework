import os
import numpy as np
import matplotlib.pyplot as plt

def plot_goodness_convergence(pos_history, neg_history, threshold1, threshold2):
    epochs = range(len(pos_history[0]))
    colors = ['r', 'g', 'b'] # 레이어 0, 1, 2 색상
    
    plt.figure(figsize=(12, 8))
    
    for i in range(len(pos_history)):
        # Positive
        plt.plot(epochs, pos_history[i], label=f'Layer {i} Positive', 
                 color=colors[i], linestyle='-', linewidth=2)
        # Negative
        plt.plot(epochs, neg_history[i], label=f'Layer {i} Negative', 
                 color=colors[i], linestyle='--', linewidth=2)
        
        # Threshold
        plt.axhline(y=threshold1[i], color=colors[i], linestyle=':', alpha=0.5)
        plt.axhline(y=threshold2[i], color=colors[i], linestyle=':', alpha=0.5)

    plt.title('Goodness Convergence per Layer (Positive vs Negative)', fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Goodness (Output Value)', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1)) # 범례를 그래프 밖에 표시
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('goodness_convergence.png')
    plt.show()
    plt.close()


def save_layer_heatmap(nets, layer_idx, epoch, save_dir='weight_heatmaps'):
    """
    특정 인덱스(layer_idx) 레이어의 가중치를 2D 히트맵으로 시각화하여 저장합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    target_net = nets[layer_idx]
    weight_tensor = None

    for name, param in target_net.named_parameters():
        if 'weight' in name:
            weight_tensor = param.data.cpu()
            break

    if weight_tensor is None:
        print(f"Layer {layer_idx}에 가중치가 없습니다.")
        return

    plt.figure(figsize=(10, 8))
    
    # 1. 4D Tensor (Convolution Layer: [Out, In, H, W])
    if weight_tensor.dim() == 4:
        out_ch, in_ch, h, w = weight_tensor.shape
        # 너무 많으면 일부(최대 64개)만 출력
        num_plots = min(out_ch, 64)
        cols = 8
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        fig.suptitle(f'Layer {layer_idx} Conv Filters (Epoch {epoch})', fontsize=16)
        
        for i in range(num_plots):
            ax = axes.flatten()[i]
            # 첫 번째 입력 채널의 필터 가중치를 보여줌
            im = ax.imshow(weight_tensor[i, 0, :, :], cmap='viridis')
            ax.axis('off')
        
        # 남는 subplot 제거
        for i in range(num_plots, rows * cols):
            axes.flatten()[i].axis('off')
            
    # 2. 2D Tensor (Linear Layer: [Out, In])
    elif weight_tensor.dim() == 2:
        plt.imshow(weight_tensor, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Layer {layer_idx} Weight Matrix (Epoch {epoch})')
        plt.xlabel('Input Features')
        plt.ylabel('Output Features')

    else:
        print(f"지원하지 않는 가중치 차원입니다: {weight_tensor.dim()}D")
        return

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'layer_{layer_idx}_heatmap_epoch_{epoch:03d}.png')
    plt.savefig(save_path)
    plt.close()
    
def save_weight_distribution(nets, epoch, save_dir='weight_plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_layers = len(nets)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
    
    # 레이어가 1개인 경우 axes가 리스트가 아니므로 예외 처리
    if num_layers == 1:
        axes = [axes]

    for i, net in enumerate(nets):
        # 각 네트워크(레이어)에서 'weight' 파라미터 추출
        weights = []
        for name, param in net.named_parameters():
            if 'weight' in name:
                weights.append(param.data.cpu().numpy().flatten())
        
        if weights:
            all_weights = np.concatenate(weights)
            axes[i].hist(all_weights, bins=50, color='teal', alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Layer {i} Weight Distribution')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(axis='y', alpha=0.3)
        else:
            axes[i].set_title(f'Layer {i} (No Weight)')

    plt.suptitle(f'Weight Distributions at Epoch {epoch}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 파일 저장
    save_path = os.path.join(save_dir, f'weights_epoch_{epoch:03d}.png')
    plt.savefig(save_path)
    plt.close() # 메모리 확보를 위해 닫기
    
def plot_loss(loss_history):
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_history)), loss_history, marker='o', color='b', label='Training Loss')
    plt.title('SCFF Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig('scff_loss_plot.png')
    plt.show()
    plt.close()