# Core API 레퍼런스

이 페이지는 핵심 `grcwa.obj` 클래스와 그 메서드를 문서화합니다.

## 클래스: `grcwa.obj`

RCWA 시뮬레이션의 메인 클래스입니다.

### 생성자

```python
grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

주기적 광학 구조에 대한 RCWA 시뮬레이션 객체를 생성합니다.

**매개변수:**

- **nG** (`int`): Fourier 전개의 목표 절단 차수
    - 실제 `nG`는 절단 방식에 따라 조정될 수 있음
    - 일반적인 값: 51-301
    - 높을수록 = 더 정확하지만 더 느림

- **L1** (`list[float, float]`): 첫 번째 격자 벡터 `[Lx1, Ly1]`
    - 첫 번째 방향의 주기성 정의
    - 단위: 선택한 길이 단위 (μm, nm 등)
    - 예: `[1.0, 0]`는 x 방향 주기 1.0

- **L2** (`list[float, float]`): 두 번째 격자 벡터 `[Lx2, Ly2]`
    - 두 번째 방향의 주기성 정의
    - L1과 직교할 필요 없음
    - 예: `[0, 1.0]`는 정사각 격자

- **freq** (`float`): 작동 주파수
    - 자연 단위에서: $f = 1/\lambda$
    - 단위: 1/(길이 단위)
    - 예: `freq=1.0`는 선택한 단위에서 $\lambda=1.0$을 의미

- **theta** (`float`): 극각 입사각 (라디안)
    - z축(표면 법선)으로부터의 각도
    - 범위: 일반적으로 $[0, \pi/2]$
    - 수직 입사: `theta=0`

- **phi** (`float`): 방위 입사각 (라디안)
    - xy평면에서 x축으로부터의 각도
    - 범위: $[0, 2\pi]$
    - 수직 입사의 경우 임의

- **verbose** (`int`, 선택적): 상세 레벨, 기본값=1
    - `0`: 조용
    - `1`: 보통 출력
    - `2`: 디버그 정보

**속성:**

초기화 후:

- `obj.nG` (`int`): 실제 사용된 절단 차수
- `obj.omega` (`complex`): 각주파수 $2\pi f$
- `obj.Layer_N` (`int`): 전체 레이어 수 (추가 후)
- `obj.G` (`ndarray`): 역격자 인덱스 `(nG, 2)`
- `obj.kx` (`ndarray`): 파동 벡터의 x 성분 `(nG,)`
- `obj.ky` (`ndarray`): 파동 벡터의 y 성분 `(nG,)`

**예:**

```python
import grcwa
import numpy as np

# 정사각 격자, 파장=1.5, 수직 입사
obj = grcwa.obj(nG=101,
                L1=[1.5, 0],
                L2=[0, 1.5],
                freq=1.0/1.5,  # λ=1.5
                theta=0,
                phi=0,
                verbose=1)

print(f"실제 nG: {obj.nG}")
print(f"각주파수: {obj.omega}")
```

---

## 레이어 메서드

### `Add_LayerUniform()`

균일한 (균질) 유전 레이어를 추가합니다.

```python
obj.Add_LayerUniform(thickness, epsilon)
```

**매개변수:**

- **thickness** (`float`): 길이 단위의 레이어 두께
    - 양수여야 함
    - 반무한 근사를 위해 매우 클 수 있음

- **epsilon** (`float` or `complex`): 상대 유전율 $\varepsilon_r$
    - 실수: 무손실 유전체
    - 복소수: 손실/흡수 매질
    - 금속의 경우 복소 유전율 또는 Drude 모델 사용
    - 예: 실리콘 $\varepsilon = 12.1$, $n = 3.48$

**반환:** None

**예:**

```python
# 진공 레이어
obj.Add_LayerUniform(1.0, 1.0)

# 실리콘 레이어 (n=3.48, ε=n²=12.1)
obj.Add_LayerUniform(0.5, 12.1)

# 손실 유전체
obj.Add_LayerUniform(0.3, 4.0 + 0.1j)

# 금속 (Drude 모델)
eps_inf = 1.0
omega_p = 9.0
gamma = 0.05
eps_metal = eps_inf - omega_p**2 / (obj.omega**2 + 1j*obj.omega*gamma)
obj.Add_LayerUniform(0.05, eps_metal)
```

---

### `Add_LayerGrid()`

데카르트 그리드에 정의된 패턴 레이어를 추가합니다.

```python
obj.Add_LayerGrid(thickness, Nx, Ny)
```

**매개변수:**

- **thickness** (`float`): 레이어 두께
- **Nx** (`int`): x 방향의 그리드 점 수
    - 일반적: 100-500
    - 높을수록 = 더 정확한 패턴 표현
- **Ny** (`int`): y 방향의 그리드 점 수

**반환:** None

**참고:**

- 패턴은 나중에 `GridLayer_geteps()`를 사용하여 제공해야 함
- 그리드는 하나의 단위 셀을 포함: $x, y \in [0, 1]$ (정규화)
- 높은 해상도는 날카로운 특징에 더 좋음
- 계산 시간은 FFT에 대해 `Nx*Ny`와 함께 스케일링됨

**예:**

```python
# 400x400 그리드의 패턴 레이어 추가
obj.Add_LayerGrid(thickness=0.3, Nx=400, Ny=400)
```

나중에, 모든 레이어가 추가된 후:

```python
# 패턴 생성
Nx, Ny = 400, 400
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# 원형 공기 홀이 있는 실리콘
eps = np.ones((Nx, Ny)) * 12.1
hole = (X-0.5)**2 + (Y-0.5)**2 < 0.4**2
eps[hole] = 1.0

# 패턴 입력
obj.GridLayer_geteps(eps.flatten())
```

---

### `Add_LayerFourier()`

해석적 Fourier 계수로 정의된 패턴 레이어를 추가합니다.

```python
obj.Add_LayerFourier(thickness, params)
```

**매개변수:**

- **thickness** (`float`): 레이어 두께
- **params**: Fourier 급수를 정의하는 매개변수
    - 형식은 구현에 따라 다름
    - 알려진 Fourier 급수를 가진 형상에 사용 (원, 직사각형)

**반환:** None

**참고:** 이 메서드는 사용 가능하지만 일반적으로 사용되지 않습니다. 그리드 메서드가 더 유연합니다.

---

## 초기화 메서드

### `Init_Setup()`

역격자를 초기화하고 균일 레이어에 대한 고유값을 계산합니다.

```python
obj.Init_Setup(Pscale=1.0, Gmethod=0)
```

**매개변수:**

- **Pscale** (`float`, 선택적): 주기 스케일링 인자, 기본값=1.0
    - 격자 벡터 스케일링: $\mathbf{L}_i \to P_{\text{scale}} \cdot \mathbf{L}_i$
    - autograd를 사용한 주기 스윕에 유용

- **Gmethod** (`int`, 선택적): 절단 방식, 기본값=0
    - `0`: 원형 절단 (등방성)
    - `1`: 직사각/평행사변형 절단

**반환:** None

**수행 작업:**

1. 역격자 벡터 $\mathbf{K}_1, \mathbf{K}_2$ 계산
2. 역격자 점 집합 $\mathbf{G}_{mn}$ 생성
3. 모든 회절 차수에 대한 파동 벡터 $\mathbf{k}_{mn}$ 계산
4. 균일 레이어에 대한 고유값 문제 풀이

**이전에 호출해야 함:**

- `GridLayer_geteps()`
- `MakeExcitationPlanewave()`
- `RT_Solve()`

**예:**

```python
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(0.5, 200, 200)
obj.Add_LayerUniform(1.0, 1.0)

# 원형 절단으로 초기화
obj.Init_Setup(Gmethod=0)

print(f"역격자 벡터: K1={obj.Lk1}, K2={obj.Lk2}")
print(f"차수 수: {obj.nG}")
```

---

## 패턴 입력 메서드

### `GridLayer_geteps()`

그리드 기반 패턴 레이어에 대한 유전 패턴 입력.

```python
obj.GridLayer_geteps(ep_all)
```

**매개변수:**

- **ep_all** (`ndarray`): 유전 상수의 평탄화된 배열
    - 1개 패턴 레이어의 경우: 형상 `(Nx*Ny,)`
    - N개 패턴 레이어의 경우: 형상 `(Nx1*Ny1 + Nx2*Ny2 + ... + NxN*NyN,)`
    - C-order (행 우선)로 평탄화되어야 함
    - 실수 또는 복소수 가능

**반환:** None

**예 (단일 레이어):**

```python
obj.Add_LayerGrid(0.3, 100, 100)
obj.Init_Setup()

# 패턴 생성
eps_grid = np.ones((100, 100)) * 4.0
# ... eps_grid 수정 ...

# 패턴 입력
obj.GridLayer_geteps(eps_grid.flatten())
```

**예 (다중 레이어):**

```python
obj.Add_LayerGrid(0.3, 100, 100)  # 레이어 1
obj.Add_LayerGrid(0.4, 150, 150)  # 레이어 2
obj.Init_Setup()

eps1 = np.ones((100, 100)) * 4.0
eps2 = np.ones((150, 150)) * 6.0
# ... 패턴 수정 ...

# 연결 및 입력
eps_all = np.concatenate([eps1.flatten(), eps2.flatten()])
obj.GridLayer_geteps(eps_all)
```

---

## 여기 메서드

### `MakeExcitationPlanewave()`

평면파 여기를 정의합니다.

```python
obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0, direction=0)
```

**매개변수:**

- **p_amp** (`float`): P-편광 진폭
    - P = TM = 입사면 내의 전기장
    - 일반적으로 0 또는 1

- **p_phase** (`float`): P-편광 위상 (라디안)
    - 보통 0

- **s_amp** (`float`): S-편광 진폭
    - S = TE = 입사면에 수직인 전기장
    - 일반적으로 0 또는 1

- **s_phase** (`float`): S-편광 위상 (라디안)
    - 원형 편광에 ±π/2 사용

- **order** (`int`, 선택적): 회절 차수 인덱스, 기본값=0
    - 보통 0 (수직 평면파 입사)
    - 특정 차수에서의 경사 입사: 다른 값 사용

- **direction** (`int`, 선택적): 입사 방향, 기본값=0
    - `0`: 위에서 입사 (입력 영역)
    - `1`: 아래에서 입사 (출력 영역)

**반환:** None

**일반적인 편광:**

```python
# P-편광
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=0, s_phase=0, order=0)

# S-편광
obj.MakeExcitationPlanewave(p_amp=0, p_phase=0,
                             s_amp=1, s_phase=0, order=0)

# 45° 선형 편광
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=0, order=0)

# 좌원 편광 (LCP)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=np.pi/2, order=0)

# 우원 편광 (RCP)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=-np.pi/2, order=0)
```

---

## 솔버 메서드

### `RT_Solve()`

반사 및 투과를 계산합니다.

```python
R, T = obj.RT_Solve(normalize=0, byorder=0)
```

**매개변수:**

- **normalize** (`int`, 선택적): 정규화 모드, 기본값=0
    - `0`: 원시 파워 (정규화 안 됨)
    - `1`: 입사 파워 및 매질 속성으로 정규화

- **byorder** (`int`, 선택적): 출력 모드, 기본값=0
    - `0`: 전체 R과 T (스칼라)
    - `1`: 차수별 R과 T (길이 `nG`의 배열)

**반환:**

- `byorder=0`인 경우: `(R, T)` (R, T는 부동소수점)
- `byorder=1`인 경우: `(Ri, Ti)` (Ri, Ti는 길이 `nG`의 배열)

**참고:**

- 물리적으로 의미 있는 결과를 위해 `normalize=1` 사용
- 무손실 구조의 경우: $R + T = 1$ (에너지 보존)
- 손실 구조의 경우: $R + T < 1$ (흡수)

**예:**

```python
# 전체 반사 및 투과
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")

# 회절 차수별
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
print(f"0차 반사: {Ri[0]:.4f}")
print(f"0차 투과: {Ti[0]:.4f}")
print(f"고차: {sum(Ri[1:]) + sum(Ti[1:]):.4f}")

# 전파되는 차수 확인
for i in range(obj.nG):
    if Ri[i] > 1e-6 or Ti[i] > 1e-6:
        m, n = obj.G[i]
        print(f"차수 ({m:2d},{n:2d}): R={Ri[i]:.4f}, T={Ti[i]:.4f}")
```

---

## 장 메서드

장 계산 메서드:

- `GetAmplitudes()`: 모드 진폭 얻기
- `Solve_FieldFourier()`: Fourier 공간에서 장 계산
- `Solve_FieldOnGrid()`: 실공간에서 장 계산
- `Volume_integral()`: 부피 적분 계산
- `Solve_ZStressTensorIntegral()`: Maxwell 응력 텐서 계산

---

## 유틸리티 메서드

### `Return_eps()`

Fourier 급수에서 유전 프로파일을 재구성합니다.

```python
eps_recon = obj.Return_eps(which_layer, Nx, Ny, component=0)
```

**매개변수:**

- **which_layer** (`int`): 레이어 인덱스
- **Nx**, **Ny** (`int`): 재구성을 위한 그리드 크기
- **component** (`int`, 선택적): 텐서 성분, 기본값=0
    - `0`: εxx (또는 스칼라 ε)
    - `1`: εyy
    - `2`: εzz

**반환:**

- **eps_recon** (`ndarray`): 재구성된 ε, 형상 `(Nx, Ny)`

**예:**

```python
# 레이어 1의 패턴 재구성
eps_recon = obj.Return_eps(which_layer=1, Nx=200, Ny=200)

import matplotlib.pyplot as plt
plt.imshow(eps_recon.T, origin='lower')
plt.colorbar(label='ε')
plt.title('재구성된 유전 패턴')
plt.show()
```

---

## 백엔드 설정

### `set_backend()`

계산 백엔드를 전환합니다.

```python
grcwa.set_backend(backend_name)
```

**매개변수:**

- **backend_name** (`str`): 사용할 백엔드
    - `'numpy'`: 표준 NumPy (더 빠름, 경사도 없음)
    - `'autograd'`: Autograd 호환 (더 느림, 경사도 지원)

**예:**

```python
import grcwa

# 최적화를 위해 autograd 사용
grcwa.set_backend('autograd')
import autograd.numpy as np
from autograd import grad

# 목적 함수 정의
def objective(epsilon):
    obj = grcwa.obj(...)
    # ... 설정 ...
    obj.GridLayer_geteps(epsilon.flatten())
    R, T = obj.RT_Solve(normalize=1)
    return -R  # 반사 최대화

# 경사도 계산
grad_obj = grad(objective)
```

**참고:** 백엔드는 동일한 스크립트에서 grcwa를 import하기 전에 설정해야 하거나, `importlib.reload()`를 사용하세요.

---

## 완전한 예제

```python
import grcwa
import numpy as np

# 설정
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.0
nG = 101

obj = grcwa.obj(nG, L1, L2, freq, theta=0, phi=0, verbose=1)

# 레이어 추가
obj.Add_LayerUniform(1.0, 1.0)        # 진공
obj.Add_LayerGrid(0.3, 200, 200)      # 패턴
obj.Add_LayerUniform(1.0, 1.0)        # 진공

# 초기화
obj.Init_Setup()

# 패턴 생성
Nx, Ny = 200, 200
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

eps = np.ones((Nx, Ny)) * 12.0
hole = (X-0.5)**2 + (Y-0.5)**2 < 0.3**2
eps[hole] = 1.0

obj.GridLayer_geteps(eps.flatten())

# 여기
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=0, s_phase=0, order=0)

# 풀이
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")

# 장 얻기
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(1, 0.15, [100, 100])
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

# 플롯
import matplotlib.pyplot as plt
plt.imshow(I.T, origin='lower', cmap='hot')
plt.colorbar(label='강도')
plt.title('패턴 레이어의 장 강도')
plt.show()
```

---

## 참고

- [튜토리얼](../tutorials/tutorial1.md): 단계별 예제
- [예제 갤러리](../examples/gallery.md): 더 많은 예제 둘러보기
- [FAQ](../reference/faq.md): 자주 묻는 질문
