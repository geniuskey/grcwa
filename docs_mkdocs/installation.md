# 설치 가이드

## 요구사항

GRCWA는 Python 3.5 이상이 필요합니다. 핵심 의존성은 다음과 같습니다:

- **numpy**: 수치 계산
- **autograd**: 자동 미분

선택적 의존성:

- **nlopt**: 최적화 예제용 (위상 최적화)
- **matplotlib**: 시각화용
- **pytest**: 테스트 실행용

## 설치 방법

### 방법 1: PyPI에서 설치 (권장)

GRCWA를 설치하는 가장 간단한 방법은 pip를 사용하는 것입니다:

```bash
pip install grcwa
```

이 명령은 numpy와 autograd를 자동으로 설치합니다.

### 방법 2: 소스에서 설치

최신 개발 버전의 경우:

```bash
# 저장소 복제
git clone https://github.com/weiliangjinca/grcwa.git
cd grcwa

# 개발 모드로 설치
pip install -e .
```

개발 모드(`-e`)를 사용하면 소스 코드를 수정할 때 변경사항이 즉시 반영됩니다.

### 방법 3: 선택적 의존성과 함께 설치

최적화 지원과 함께 설치하려면:

```bash
pip install grcwa nlopt matplotlib
```

## 설치 확인

설치를 테스트하세요:

```python
import grcwa
import numpy as np

print(f"GRCWA를 성공적으로 가져왔습니다!")
print(f"NumPy 버전: {np.__version__}")

# 백엔드 테스트
grcwa.set_backend('numpy')
print("NumPy 백엔드: 정상")

grcwa.set_backend('autograd')
print("Autograd 백엔드: 정상")
```

간단한 시뮬레이션 실행:

```python
import grcwa
import numpy as np

# 간단한 3층 구조
obj = grcwa.obj(nG=101, L1=[1,0], L2=[0,1],
                freq=1.0, theta=0, phi=0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)  # 진공
obj.Add_LayerUniform(0.5, 4.0)  # 유전체 슬랩
obj.Add_LayerUniform(1.0, 1.0)  # 진공
obj.Init_Setup()

# 여기
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0, order=0)

# 풀기
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")
```

예상 출력:
```
R = 0.3600, T = 0.6400, R+T = 1.0000
```

## 개발 환경 설정

GRCWA에 기여하거나 수정할 계획이라면:

### 1. 가상 환경 생성

```bash
# venv 사용
python -m venv grcwa_env
source grcwa_env/bin/activate  # Windows: grcwa_env\Scripts\activate

# conda 사용
conda create -n grcwa python=3.8
conda activate grcwa
```

### 2. 개발 의존성 설치

```bash
pip install -e ".[dev]"
```

또는 수동으로:

```bash
pip install pytest pytest-cov flake8 sphinx
```

### 3. 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트 실행
pytest tests/test_rcwa.py::test_rcwa

# 커버리지와 함께 실행
pytest --cov=grcwa tests/
```

## 백엔드 구성

GRCWA는 두 가지 계산 백엔드를 지원합니다:

### NumPy 백엔드 (기본)

표준 NumPy 연산 - 빠르지만 자동 미분 없음:

```python
import grcwa
grcwa.set_backend('numpy')
```

다음의 경우 사용:
- 순방향 시뮬레이션만 필요 (R, T, 장)
- 최대 성능 필요
- 경사도가 필요하지 않음

### Autograd 백엔드

Autograd 호환 NumPy - 자동 미분 지원:

```python
import grcwa
grcwa.set_backend('autograd')
```

다음의 경우 사용:
- 최적화를 위한 경사도 필요
- 역설계 수행
- 매개변수 민감도 계산

!!! warning "백엔드 호환성"
    autograd 백엔드로 전환하면, 모든 배열을 표준 numpy가 아닌 autograd.numpy로 생성해야 합니다:

    ```python
    import autograd.numpy as np  # 이것 사용
    # 사용 금지: import numpy as np
    ```

## 일반적인 설치 문제

### 문제: "No module named autograd"

**해결**: autograd 설치:
```bash
pip install autograd
```

### 문제: Numpy 버전 충돌

**해결**: 호환되는 numpy 버전 확인:
```bash
pip install --upgrade numpy autograd
```

### 문제: "ImportError: cannot import name 'obj'"

**해결**: 패키지에서 가져오는지 확인:
```python
import grcwa
obj = grcwa.obj(...)  # 올바름

# 사용 금지:
from grcwa import obj  # 이전 버전에서는 작동하지 않을 수 있음
```

### 문제: 테스트가 "ModuleNotFoundError"로 실패

**해결**: 테스트 의존성 설치:
```bash
pip install pytest
```

### 문제: NLOPT를 사용할 수 없음

NLOPT는 선택사항이며 최적화 예제에만 필요합니다.

**Linux**:
```bash
pip install nlopt
```

**Mac** (Homebrew 사용):
```bash
brew install nlopt
pip install nlopt
```

**Windows**: conda 사용:
```bash
conda install -c conda-forge nlopt
```

## Docker 설치 (선택사항)

재현 가능한 환경을 위해:

`Dockerfile` 생성:
```dockerfile
FROM python:3.8-slim

RUN pip install grcwa nlopt matplotlib jupyter

WORKDIR /workspace

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

빌드 및 실행:
```bash
docker build -t grcwa .
docker run -p 8888:8888 -v $(pwd):/workspace grcwa
```

## 플랫폼별 참고사항

### Linux
pip로 바로 작동합니다.

### macOS
Xcode 명령줄 도구가 필요할 수 있습니다:
```bash
xcode-select --install
```

### Windows
표준 Python 설치로 작동합니다. Anaconda를 사용하는 경우:
```bash
conda install numpy autograd
pip install grcwa
```

## 성능 최적화

### MKL을 사용한 NumPy

더 나은 성능을 위해 Intel MKL로 빌드된 NumPy 사용:

```bash
# conda 사용
conda install numpy scipy mkl

# 확인
python -c "import numpy; numpy.show_config()"
```

### 병렬 계산

GRCWA 자체는 멀티스레딩을 사용하지 않지만, 매개변수 스윕을 병렬화할 수 있습니다:

```python
from multiprocessing import Pool

def compute_spectrum(freq):
    obj = grcwa.obj(...)
    # ... 설정 ...
    R, T = obj.RT_Solve()
    return R, T

freqs = np.linspace(0.5, 1.5, 50)
with Pool(8) as p:
    results = p.map(compute_spectrum, freqs)
```

## 다음 단계

이제 GRCWA가 설치되었습니다:

1. **[빠른 시작 가이드](quickstart.md)**: 첫 번째 시뮬레이션 실행
2. **[기본 개념](guide/concepts.md)**: 작업 흐름 이해
3. **[예제](examples/gallery.md)**: 예제 시뮬레이션 탐색
4. **[튜토리얼](tutorials/tutorial1.md)**: 단계별 학습

## 도움 받기

문제가 발생하면:

- [FAQ](reference/faq.md) 읽기
- [GitHub](https://github.com/weiliangjinca/grcwa/issues)에 이슈 열기
- 연락처: jwlaaa@gmail.com
