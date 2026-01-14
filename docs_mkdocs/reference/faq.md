# 자주 묻는 질문 (FAQ)

## 일반 질문

### RCWA란 무엇인가요?

엄밀 결합파 해석(RCWA)은 주기 구조에서 Maxwell 방정식을 푸는 준해석적 방법입니다. 전자기장을 Fourier 급수로 전개하고 결과 고유값 문제를 레이어별로 풉니다. RCWA는 (수치 정밀도 내에서) 정확하며 주기적 광결정 구조에 특히 효율적입니다.

### FDTD나 FEM 대신 RCWA를 언제 사용해야 하나요?

**RCWA 사용:**

- 구조가 2D 주기성을 가질 때 (광결정, 격자, 메타표면)
- 스펙트럼 또는 각도 응답이 필요할 때 (주파수/각도 스윕)
- 주기 구조의 빠른 시뮬레이션을 원할 때
- 원거리장 회절 패턴이 필요할 때

**FDTD/FEM 사용:**

- 구조가 비주기적일 때 (고립 물체, 랜덤 구조)
- 시간 영역 응답이 필요할 때
- 주기성 없는 3D 임의 기하학
- 단일 실행의 광대역 시뮬레이션 (FDTD의 장점)

### "autoGradable"은 무엇을 의미하나요?

GRCWA는 [Autograd](https://github.com/HIPS/autograd)와 통합되어 자동 미분을 가능하게 합니다. 이는 수동으로 수반 방정식을 유도하지 않고도 모든 입력(ε, 주파수, 각도, 두께)에 대한 모든 출력(R, T, 장)의 경사도를 자동으로 계산할 수 있음을 의미합니다. 이는 다음에 필수적입니다:

- 위상 최적화
- 역설계
- 민감도 분석
- 경사도 기반 최적화

## 설치 및 설정

### GRCWA를 어떻게 설치하나요?

```bash
pip install grcwa
```

최신 개발 버전의 경우:

```bash
git clone https://github.com/weiliangjinca/grcwa
cd grcwa
pip install -e .
```

### 의존성은 무엇인가요?

**필수:**

- Python ≥ 3.5
- numpy
- autograd

**선택:**

- nlopt (최적화 예제용)
- matplotlib (시각화용)
- pytest (테스트용)

### NumPy와 Autograd 백엔드 간에 어떻게 전환하나요?

```python
import grcwa

# NumPy 백엔드 (더 빠름, 경사도 없음)
grcwa.set_backend('numpy')

# Autograd 백엔드 (경사도 활성화)
grcwa.set_backend('autograd')
```

**중요:** `grcwa.obj` 인스턴스를 생성하기 전에 백엔드를 설정하세요.

## 사용 질문

### 절단 차수(nG)를 어떻게 선택하나요?

**경험 법칙:**

- 균일 레이어: `nG = 11-51`
- 매끄러운 패턴: `nG = 51-101`
- 날카로운 특징: `nG = 101-301`
- 매우 미세한 디테일: `nG = 301-501`

**항상 수렴을 테스트하세요:**

```python
for nG in [51, 101, 201, 301]:
    obj = grcwa.obj(nG, ...)
    # ... 설정 및 풀이 ...
    R, T = obj.RT_Solve(normalize=1)
    print(f"nG={obj.nG}: R={R:.6f}, T={T:.6f}")
```

### 그리드 점(Nx, Ny)을 얼마나 사용해야 하나요?

**권장사항:**

- 매끄러운 특징: 50-100
- 일반적인 패턴: 200-400
- 날카로운 가장자리: 400-500
- 매우 미세한 디테일: 500-1000

**트레이드오프:** 더 높은 해상도 = 더 정확하지만 더 느린 FFT.

### 어떤 단위를 사용해야 하나요?

GRCWA는 $c = \varepsilon_0 = \mu_0 = 1$인 **자연 단위**를 사용합니다. 일관된 길이 단위를 사용할 수 있습니다:

```python
# 예: 모두 μm 단위
wavelength = 1.55  # μm
freq = 1.0 / wavelength  # ≈ 0.645
L1 = [0.5, 0]  # μm
thickness = 0.3  # μm
```

**핵심:** 모든 길이는 **동일한 단위**를 사용해야 합니다.

### 파장을 주파수로 어떻게 변환하나요?

자연 단위에서:

$$
f = \frac{c}{\lambda} = \frac{1}{\lambda}
$$

```python
wavelength = 1.5  # 선택한 단위
freq = 1.0 / wavelength
```

### P 편광과 S 편광의 차이는 무엇인가요?

**P-편광 (TM):**

- 입사면 내의 전기장
- 입사면에 수직인 자기장
- `p_amp=1, s_amp=0` 설정

**S-편광 (TE):**

- 입사면에 수직인 전기장
- 입사면 내의 자기장
- `p_amp=0, s_amp=1` 설정

등방성 재료에 대한 수직 입사의 경우 P와 S는 동일한 결과를 제공합니다.

### 원형 편광을 어떻게 정의하나요?

**좌원 편광 (LCP):**

```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=np.pi/2, order=0)
```

**우원 편광 (RCP):**

```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=-np.pi/2, order=0)
```

## 문제 해결

### R + T ≠ 1인 이유는 무엇인가요?

**가능한 원인:**

1. **불충분한 절단 차수:** `nG` 증가
2. **수치 불안정성:** 레이어 두께 감소 또는 더 작은 `nG` 사용
3. **흡수 재료:** 손실 재료의 경우 $R + T < 1$이 정확함 (흡수 = $1-R-T$)
4. **매우 높은 대비:** 더 많은 그리드 점 사용 (`Nx, Ny`)

**수정:**

```python
# 수렴 테스트
for nG in [101, 201, 301, 501]:
    obj = grcwa.obj(nG, ...)
    # ... 풀기 ...
    R, T = obj.RT_Solve(normalize=1)
    print(f"nG={nG}: R+T={R+T:.6f}, error={(R+T-1):.2e}")
```

### 특이 행렬 오류가 발생하는 이유는 무엇인가요?

**원인:**

- 손실 없는 완전 수직 입사
- 완전 대칭 구조

**수정:** 작은 손실 추가:

```python
Qabs = 1e8  # 매우 높은 Q-팩터
freq = freq * (1 + 1j / (2*Qabs))
```

또는 epsilon에 작은 허수부 추가:

```python
epsilon = 4.0 + 1e-10j
```

### 내 패턴이 잘못 보입니다. 어떻게 디버그하나요?

`Return_eps()`를 사용하여 시각화:

```python
eps_recon = obj.Return_eps(which_layer=1, Nx=200, Ny=200)

import matplotlib.pyplot as plt
plt.imshow(eps_recon.T, origin='lower')
plt.colorbar()
plt.title('재구성된 유전 패턴')
plt.show()
```

정확성을 확인하기 위해 입력 패턴과 비교하세요.

### 결과가 매우 느립니다. 어떻게 속도를 높이나요?

**최적화 전략:**

1. **절단 차수 감소:** 먼저 더 낮은 `nG` 시도
2. **NumPy 백엔드 사용:** 경사도가 필요하지 않으면 Autograd보다 빠름
3. **그리드 해상도 감소:** 허용 가능하면 더 낮은 `Nx, Ny`
4. **가능한 경우 균일 레이어 사용:** 패턴보다 훨씬 빠름
5. **매개변수 스윕 병렬화:**

```python
from multiprocessing import Pool

def compute(freq):
    obj = grcwa.obj(...)
    # ... 풀기 ...
    return R, T

with Pool(8) as p:
    results = p.map(compute, frequencies)
```

### 이방성 재료를 어떻게 처리하나요?

이방성 유전 텐서의 경우 3개 성분의 리스트 제공:

```python
# εxx = εyy ≠ εzz인 일축 재료
eps_xx = 4.0
eps_yy = 4.0
eps_zz = 6.0

eps_tensor = [eps_xx, eps_yy, eps_zz]

# 그리드 레이어의 경우
eps_grid_xx = np.ones((Nx, Ny)) * eps_xx
eps_grid_yy = np.ones((Nx, Ny)) * eps_yy
eps_grid_zz = np.ones((Nx, Ny)) * eps_zz

eps_all = [eps_grid_xx.flatten(), eps_grid_yy.flatten(), eps_grid_zz.flatten()]
obj.GridLayer_geteps(eps_all)
```

## 고급 사용

### 위상 최적화를 어떻게 하나요?

```python
import grcwa
grcwa.set_backend('autograd')
import autograd.numpy as np
from autograd import grad

def objective(epsilon):
    obj = grcwa.obj(101, [1,0], [0,1], freq, 0, 0, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Add_LayerGrid(0.5, Nx, Ny)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Init_Setup()
    obj.GridLayer_geteps(epsilon.flatten())
    obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R, T = obj.RT_Solve(normalize=1)
    return -R  # 반사 최대화

# 경사도 계산
grad_obj = grad(objective)

# 최적화기 사용 (예: NLOPT, scipy.optimize)
import nlopt

def nlopt_objective(x, grad_array):
    if grad_array.size > 0:
        grad_array[:] = grad_obj(x)
    return objective(x)

# NLOPT 설정
opt = nlopt.opt(nlopt.LD_MMA, Nx*Ny)
opt.set_min_objective(nlopt_objective)
opt.set_lower_bounds(1.0)
opt.set_upper_bounds(12.0)

# 초기 추정
epsilon_init = np.ones(Nx*Ny) * 6.0

# 최적화
epsilon_opt = opt.optimize(epsilon_init)
```

### 고립된 물체를 시뮬레이션할 수 있나요?

RCWA는 주기성을 요구합니다. 고립 물체의 경우:

1. **큰 단위 셀 사용** (슈퍼셀 접근):

```python
# 물체 크기: 1 μm
# 10 μm × 10 μm 단위 셀 사용
L1 = [10, 0]
L2 = [0, 10]

# 중앙에 물체 배치, 진공으로 둘러싸기
```

2. **수렴 확인**: 셀 크기를 증가시키며 결과가 안정될 때까지

### 반사/투과 계수를 어떻게 얻나요 (파워만이 아닌)?

진폭은 내부에 저장됩니다. 다음을 통해 액세스:

```python
# 풀이 후
amplitudes_in, amplitudes_out = obj.GetAmplitudes(layer=0, z_offset=0)
```

반사 계수의 경우:

```python
# 각 차수의 반사 계수
r_coeffs = amplitudes_out[0:obj.nG]  # 입력 영역에서 후방 전파
```

### 흡수를 어떻게 계산하나요?

손실 재료의 경우:

```python
R, T = obj.RT_Solve(normalize=1)
A = 1 - R - T  # 흡수
```

또는 $\text{Im}(\varepsilon)|E|^2$의 부피 적분 사용:

```python
absorption = obj.Volume_integral(which_layer, Mx, My, Mz, normalize=1)
```

### 3D 광결정에 GRCWA를 사용할 수 있나요?

RCWA는 2D (xy-평면)의 주기성과 z의 적층을 가정합니다. 3D 광결정의 경우:

- ✅ **광결정 슬랩** (2D 주기, z에서 유한): 예
- ❌ **3D 벌크 광결정** (xyz에서 주기): 아니오, 평면파 전개 또는 다른 방법 사용

### GRCWA를 어떻게 인용하나요?

```bibtex
@article{Jin2020,
  title = {Inverse design of lightweight broadband reflector for relativistic lightsail propulsion},
  author = {Jin, Weiliang and Li, Wei and Orenstein, Meir and Fan, Shanhui},
  journal = {ACS Photonics},
  volume = {7},
  number = {9},
  pages = {2350--2355},
  year = {2020},
  publisher = {ACS Publications}
}
```

## 일반적인 오류 메시지

### "IndexError: index out of range"

**원인:** 패턴 레이어 수와 제공된 epsilon 배열 간의 불일치.

**수정:** `epsilon.flatten()`의 총 길이가 올바른지 확인.

### "ValueError: operands could not be broadcast together"

**원인:** 패턴 정의에서 배열 형상 불일치.

**수정:** 패턴의 형상이 `(Nx, Ny)`이고 meshgrid에서 `indexing='ij'` 사용 확인.

### "LinAlgError: Singular matrix"

**원인:** 수치 특이점, 종종 손실 없는 수직 입사에서.

**수정:** 작은 손실 추가:

```python
freq = freq * (1 + 1e-10j)
```

## 여전히 질문이 있으신가요?

- [튜토리얼](../tutorials/tutorial1.md) 읽기
- [예제](../examples/gallery.md) 둘러보기
- [GitHub](https://github.com/weiliangjinca/grcwa/issues)에 이슈 열기
- 연락처: jwlaaa@gmail.com
