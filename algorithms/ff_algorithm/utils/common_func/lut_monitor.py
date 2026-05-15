import numpy as np
import matplotlib.pyplot as plt
import torch

class LUTErrorMonitor:
    def __init__(self):
        self.enabled = False
        self.errors = {
            'reciprocal': [], 'sqrt': [], 'exp': [], 'log': [], 'pow': []
        }
        self.data = {
            'ideal': [], 'lut': []
        }
        
    def enable(self):
        self.enabled = True
        print("[LUT Monitor] Error tracking enabled.")

    def disable(self):
        self.enabled = False
        print("[LUT Monitor] Error tracking disabled.")

    def reset(self):
        self.errors = {k: [] for k in self.errors}
        
    def update(self, func_name, ideal_tensor, lut_tensor):
        """
        매 연산마다 Ideal 값과 LUT 값의 차이(Mean Absolute Error)를 기록
        """
        if not self.enabled:
            return
        
        with torch.no_grad():
            # FP16 오버플로우 방지를 위해 float32로 변환 후 오차 계산
            # diff = torch.abs(ideal_tensor.float() - lut_tensor.float())
            diff = torch.abs((ideal_tensor.float() - lut_tensor.float()) / (ideal_tensor.float() + 1e-4))
            
            mae = diff.mean().item() 
            self.errors[func_name].append(mae)
            # if func_name == 'sqrt':
            #     self.data['ideal'].append(ideal_tensor)
            #     self.data['lut'].append(lut_tensor)

    def plot_errors(self, save_path=None):
        """
        누적된 오차의 분포를 히스토그램으로 출력
        X축: Error 값
        Y축: Count (빈도)
        """
        # 데이터가 하나라도 있는지 확인
        if not any(len(v) > 0 for v in self.errors.values()):
            print("[LUT Monitor] No error data collected to plot.")
            return

        plt.figure(figsize=(10, 6))
        
        # 각 함수별로 히스토그램 그리기
        for name, err_list in self.errors.items():
            if len(err_list) > 0:
                # bins: 계급 개수 (조절 가능)
                # alpha: 투명도 (겹쳐서 보이게 함)
                # log: Y축(Count)을 로그로 표시할지 여부
                plt.hist(err_list, bins=100, alpha=0.5, label=f'{name}', log=True)

        plt.title("LUT Approximation Error Distribution")
        plt.xlabel("Mean Absolute Error (Ideal - LUT)")
        plt.ylabel("Frequency / Count (Log Scale)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"[LUT Monitor] Plot saved to {save_path}")
        else:
            plt.show()
