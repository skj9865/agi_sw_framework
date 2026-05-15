# Unified SW Framework

Brain-inspired algorithm integration framework for KETI Global R&D.

Forward-Forward (SCFF), TBP Monty, Neuromorphic 등 여러 알고리즘을 하나의 Python 환경에서 실행하고 비교할 수 있다.

## Quick Start

### 1. 환경 설치

```bash
bash install.sh
```

Python 3.10이 PATH에 없는 경우 `install.sh` 상단의 `PYTHON=` 주석을 풀고 경로를 직접 지정한다.

### 2. 활성화

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 3. 실행

```bash
# 등록된 알고리즘 목록
python run.py --list

# 단일 알고리즘 실행
python run.py --algorithm ff --dataset cifar10 --mode train
python run.py --algorithm ff --dataset cifar10 --mode evaluate
python run.py --algorithm monty --mode evaluate

# 여러 알고리즘 비교
python run.py --compare ff monty --mode evaluate
```

## 프로젝트 구조

```
SW_framework/
├── run.py                  # CLI 진입점
├── install.sh              # 원클릭 환경 설치
├── requirements.txt        # 통합 의존성
│
├── config/
│   └── framework_config.yaml   # 프레임워크 설정
│
├── core/                   # 프레임워크 핵심
│   ├── base_algorithm.py   # 알고리즘 추상 인터페이스
│   ├── registry.py         # 알고리즘 등록/조회
│   └── runner.py           # 실행 엔진 + 비교 기능
│
├── algorithms/             # 알고리즘 구현
│   ├── __init__.py         # wrapper import (등록 트리거)
│   ├── ff_algorithm/       # Forward-Forward
│   │   ├── wrapper.py      # BaseAlgorithm 구현
│   │   └── ...             # 기존 FF 코드 (수정 없음)
│   └── monty/              # TBP Monty
│       ├── wrapper/
│       │   └── monty_algorithm.py  # BaseAlgorithm 구현
│       ├── scripts/
│       │   └── monty_inference.py
│       └── tbp.monty/      # Monty 소스
│
├── dataset/                # 데이터셋 (git 제외)
├── model/                  # 사전학습 모델 (git 제외)
└── results/                # 실행 결과 (git 제외)
```

## 새 알고리즘 추가 방법

3단계로 새 알고리즘을 통합할 수 있다.

### Step 1. wrapper 작성

`algorithms/<알고리즘명>/wrapper.py`를 만들고, `BaseAlgorithm`을 상속하여 5개 메서드를 구현한다.

```python
import sys, os

_ALGO_DIR = os.path.dirname(os.path.abspath(__file__))

from core.base_algorithm import BaseAlgorithm
from core.registry import register_algorithm


@register_algorithm
class MyAlgorithm(BaseAlgorithm):

    def name(self) -> str:
        return "myalgo"                     # run.py --algorithm myalgo

    def configure(self, config: dict) -> None:
        self._dataset = config.get("dataset", config.get("default_dataset"))
        self._seed = config.get("seed", 42)
        # config는 framework + algorithms.myalgo 설정이 합쳐져서 들어온다

    def train(self, **kwargs) -> dict:
        # 학습 로직 (기존 코드 호출)
        return {"accuracy": 0.95, "dataset": self._dataset}

    def evaluate(self, **kwargs) -> dict:
        # 평가 로직
        return {"accuracy": 0.93, "dataset": self._dataset}

    def get_supported_datasets(self) -> list:
        return ["cifar10", "mnist"]
```

**핵심 규칙:**
- `@register_algorithm` 데코레이터 필수
- `name()`이 반환하는 문자열이 CLI에서 사용하는 알고리즘 식별자
- `train()`과 `evaluate()`는 반드시 `dict`를 반환 (최소 `accuracy` 키 포함)
- 기존 코드가 `argparse`를 사용한다면, 호출 전에 `sys.argv`를 격리해야 한다:
  ```python
  saved_argv = sys.argv
  sys.argv = [sys.argv[0]]
  try:
      # 기존 코드 호출
  finally:
      sys.argv = saved_argv
  ```
- 기존 코드가 `plt.show()`를 사용한다면, wrapper 상단에 추가:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  ```

### Step 2. algorithms/\_\_init\_\_.py 에 import 추가

```python
import algorithms.ff_algorithm.wrapper
import algorithms.monty.wrapper.monty_algorithm
import algorithms.myalgo.wrapper            # <-- 추가
```

이 import가 있어야 `@register_algorithm`이 실행되어 알고리즘이 등록된다.

### Step 3. framework_config.yaml 에 설정 추가

```yaml
algorithms:
  ff:
    ...
  monty:
    ...
  myalgo:                          # <-- 추가
    enabled: true
    default_dataset: "cifar10"
    # 알고리즘별 추가 설정
```

### 확인

```bash
python run.py --list
# 출력:
#   - ff           datasets: cifar10, mnist, svhn
#   - monty        datasets: world_image
#   - myalgo       datasets: cifar10, mnist        <-- 새로 추가됨
```

## 설정

`config/framework_config.yaml`에서 전체 프레임워크와 알고리즘별 설정을 관리한다.

```yaml
framework:
  device: "cuda:0"          # GPU 디바이스 (Monty는 CPU only)
  seed: 1234                # 랜덤 시드
  results_dir: "./results"  # 결과 저장 경로

algorithms:
  ff:
    enabled: true
    default_dataset: "cifar"
    batchsize: 100
  monty:
    enabled: true
    data_path: "dataset/worldimages/standard_scenes"
    model_path: "model/monty/.../pretrained"
    max_episodes: null      # null -> 전체 48 에피소드
    max_eval_steps: 500
```

## 다른 환경에서 실행

### 필요 사항
- Python 3.10
- CUDA 11.8 (GPU 사용 시)

### 설치
```bash
git clone https://github.com/skj9865/agi_sw_framework.git
cd agi_sw_framework
bash install.sh
```

### Monty 실행 시 추가 준비
FF는 데이터셋을 자동 다운로드하지만, Monty는 수동으로 준비해야 한다:
- `dataset/worldimages/standard_scenes/` - world image 데이터셋
- `model/monty/.../pretrained/model.pt` - 사전학습된 모델

### CPU only 환경
`install.sh`에서 PyTorch 설치 줄의 `cu118`을 `cpu`로 변경:
```bash
"$PIP" install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

## 통합 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| torch | 2.0.0 | FF + Monty 공통 |
| numpy | 1.23.5 | FF + Monty 공통 |
| scipy | 1.15.3 | FF + Monty 공통 |
| matplotlib | 3.7.3 | FF + Monty 공통 |
| torch-geometric | 2.7.0+ | Monty 모델 로딩 |
| pyyaml | 6.0+ | 프레임워크 설정 |
