# 변수 및 규약 레퍼런스

이 페이지는 GRCWA에서 사용되는 모든 중요한 변수, 의미 및 규약을 문서화합니다.

## 물리 상수 및 단위

### 자연 단위

GRCWA는 다음과 같은 **자연 단위**를 사용합니다:

| 상수 | 값 | 의미 |
|----------|-------|---------|
| $\varepsilon_0$ | 1 | 진공 유전율 |
| $\mu_0$ | 1 | 진공 투자율 |
| $c$ | 1 | 빛의 속도 |
| $Z_0 = \sqrt{\mu_0/\varepsilon_0}$ | 1 | 진공 임피던스 |

**의미:**

- 임의의 길이 단위 선택 가능 (μm, nm, mm 등)
- 주파수 $f = 1/\lambda$ (선택한 단위)
- 유전 상수는 무차원: $\varepsilon_r = \varepsilon/\varepsilon_0$
- $c=1$로 모든 방정식 단순화

### 예: 파장 1.55 μm

```python
wavelength = 1.55  # μm
freq = 1.0 / wavelength  # freq ≈ 0.645 자연 단위
L1 = [0.5, 0]  # 격자 상수 0.5 μm
thickness = 0.3  # 레이어 두께 0.3 μm
```

모든 길이는 **동일한 단위**(이 예에서는 μm)를 사용해야 합니다.

## 시간 조화 규약

장은 다음과 같이 진동합니다:

$$
\mathbf{E}(\mathbf{r}, t) = \text{Re}[\mathbf{E}(\mathbf{r}) e^{-i\omega t}]
$$

**규약**: $e^{-i\omega t}$ ($e^{+i\omega t}$가 아님)

**의미:**

- 공간의 위상 진행: 전방 전파의 경우 $e^{+ikz}$
- 흡수: $\varepsilon = \varepsilon' + i\varepsilon''$이고 $\varepsilon'' > 0$
- 감쇠파 감소: $e^{-\kappa z}$이고 $\kappa > 0$

## 핵심 클래스: `grcwa.obj`

### 생성자 매개변수

```python
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

| 매개변수 | 타입 | 단위 | 설명 |
|-----------|------|-------|-------------|
| `nG` | int | - | 목표 Fourier 고조파 수 |
| `L1` | list[float, float] | length | 격자 벡터 1: `[Lx1, Ly1]` |
| `L2` | list[float, float] | length | 격자 벡터 2: `[Lx2, Ly2]` |
| `freq` | float | 1/length | 주파수 ($\omega/(2\pi c)$) |
| `theta` | float | radians | 극각 입사각 (z축으로부터) |
| `phi` | float | radians | 방위각 (xy평면에서 x축으로부터) |
| `verbose` | int | - | 상세 레벨: 0 (조용), 1 (보통), 2 (디버그) |

**참고:**

- 절단 방식에 따라 실제 `nG`가 다를 수 있음
- `L1`, `L2`는 직교할 필요 없음 (경사 격자 지원)
- 수직 입사: `theta=0`, `phi=0`
- 경사 입사: $0 < \theta < \pi/2$

### 격자 벡터

```python
L1 = [Lx1, Ly1]  # 첫 번째 격자 벡터
L2 = [Lx2, Ly2]  # 두 번째 격자 벡터
```

**일반적인 격자:**

**정사각:**
```python
a = 1.0
L1 = [a, 0]
L2 = [0, a]
```

**직사각:**
```python
a, b = 1.0, 0.5
L1 = [a, 0]
L2 = [0, b]
```

**육각:**
```python
a = 1.0
L1 = [a, 0]
L2 = [a/2, a*np.sqrt(3)/2]  # 60° 각도
```

**능면체:**
```python
a = 1.0
angle = 75 * np.pi/180
L1 = [a, 0]
L2 = [a*np.cos(angle), a*np.sin(angle)]
```

### 각도

**Theta** ($\theta$): z축으로부터의 극각

- $\theta = 0$: 수직 입사
- $0 < \theta < \pi/2$: 위에서 경사 입사
- $\pi/2 < \theta < \pi$: 스침 입사 (거의 사용 안 함)

**Phi** ($\phi$): xy평면의 방위각

- $\phi = 0$: xz평면에서 입사
- $\phi = \pi/2$: yz평면에서 입사
- 일반: 입사 방향은 $(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$

**파동 벡터:**

$$
\mathbf{k}_{\text{inc}} = \omega\sqrt{\varepsilon_{\text{in}}}(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)
$$

## 레이어 변수

### 레이어 타입

| 타입 ID | 메서드 | 설명 |
|---------|--------|-------------|
| 0 | `Add_LayerUniform` | 균질 유전체 |
| 1 | `Add_LayerGrid` | 데카르트 그리드에 정의된 패턴 |
| 2 | `Add_LayerFourier` | Fourier 계수로 정의된 패턴 |

### 균일 레이어

```python
obj.Add_LayerUniform(thickness, epsilon)
```

| 매개변수 | 타입 | 단위 | 설명 |
|-----------|------|-------|-------------|
| `thickness` | float | length | 레이어 두께 |
| `epsilon` | float or complex | - | 상대 유전율 $\varepsilon_r$ |

**예:**

```python
# 진공
obj.Add_LayerUniform(1.0, 1.0)

# 실리콘 (n=3.48)
obj.Add_LayerUniform(0.5, 3.48**2)  # ε = n²

# 손실이 있는 실리콘
obj.Add_LayerUniform(0.5, 12.1 + 0.1j)  # ε = ε' + iε''

# 은 (Drude 모델)
eps_inf = 5.0
omega_p = 9.0  # 플라즈마 주파수
gamma = 0.02  # 감쇠
eps_Ag = eps_inf - omega_p**2 / (freq**2 + 1j*freq*gamma)
obj.Add_LayerUniform(0.02, eps_Ag)
```

### 패턴 레이어 (그리드)

```python
obj.Add_LayerGrid(thickness, Nx, Ny)
```

| 매개변수 | 타입 | 단위 | 설명 |
|-----------|------|-------|-------------|
| `thickness` | float | length | 레이어 두께 |
| `Nx` | int | - | x 방향 그리드 점 수 |
| `Ny` | int | - | y 방향 그리드 점 수 |

나중에 패턴 입력:

```python
epsilon_grid = ...  # 형상: (Nx, Ny)
obj.GridLayer_geteps(epsilon_grid.flatten())
```

**패턴 좌표:**

- $i=0,\ldots,N_x-1$, $j=0,\ldots,N_y-1$에 대해 $(i/(N_x-1), j/(N_y-1))$의 그리드 점
- 정규화된 좌표 $x, y \in [0, 1]$
- 물리 좌표: 격자 벡터로 곱하기

**다중 패턴 레이어:**

여러 패턴 레이어가 있는 경우 평탄화하고 연결:

```python
epsilon1 = ...  # (Nx1, Ny1)
epsilon2 = ...  # (Nx2, Ny2)
epsilon_all = np.concatenate([epsilon1.flatten(), epsilon2.flatten()])
obj.GridLayer_geteps(epsilon_all)
```

### 그리드 해상도

**권장사항:**

| 특징 크기 | 권장 Nx, Ny |
|--------------|-------------------|
| 부드러운 변화 | 50-100 |
| 날카로운 특징 | 200-400 |
| 매우 미세한 디테일 | 500-1000 |

**트레이드오프:** 높은 해상도 → 더 정확하지만 더 느린 FFT.

## Fourier 고조파

### 절단 차수

```python
nG = 101  # 목표 수
```

초기화 후 실제 `nG`:

```python
obj.nG  # 실제 사용된 수
obj.G   # (m, n) 인덱스 배열, 형상: (nG, 2)
```

**절단 방식:**

- **원형** (`Gmethod=0`, 기본값): $m^2 + n^2 \leq N_{\max}^2$
- **직사각** (`Gmethod=1`): $|m| \leq M$, $|n| \leq N$

**수렴:**

```python
# 수렴 테스트
nG_values = [51, 101, 201, 301, 501]
for nG in nG_values:
    obj = grcwa.obj(nG, ...)
    # ... 풀기 ...
    R, T = obj.RT_Solve()
    print(f"nG={obj.nG}, R={R:.6f}, T={T:.6f}")
```

`nG`를 증가시켜도 R, T가 크게 변하지 않으면 수렴됨.

## 역격자

`Init_Setup()` 후:

| 속성 | 타입 | 설명 |
|-----------|------|-------------|
| `obj.Lk1` | array | 역격자 벡터 1 |
| `obj.Lk2` | array | 역격자 벡터 2 |
| `obj.G` | array (nG, 2) | $(m, n)$ 인덱스 배열 |
| `obj.kx` | array (nG,) | 파동 벡터의 x 성분 |
| `obj.ky` | array (nG,) | 파동 벡터의 y 성분 |

**역격자 벡터:**

$$
\mathbf{K}_1 \cdot \mathbf{L}_1 = 2\pi, \quad \mathbf{K}_1 \cdot \mathbf{L}_2 = 0
$$

$$
\mathbf{K}_2 \cdot \mathbf{L}_1 = 0, \quad \mathbf{K}_2 \cdot \mathbf{L}_2 = 2\pi
$$

**각 차수의 파동 벡터:**

$$
k_{x,mn} = k_{x0} + m K_{1x} + n K_{2x}
$$

$$
k_{y,mn} = k_{y0} + m K_{1y} + n K_{2y}
$$

코드에서 액세스:

```python
kx_mn = obj.kx  # 형상: (nG,)
ky_mn = obj.ky  # 형상: (nG,)
G_indices = obj.G  # 형상: (nG, 2), 각 행은 (m, n)
```

## 여기 변수

### 평면파 여기

```python
obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order, direction=0)
```

| 매개변수 | 타입 | 단위 | 설명 |
|-----------|------|-------|-------------|
| `p_amp` | float | - | P-편광 진폭 |
| `p_phase` | float | radians | P-편광 위상 |
| `s_amp` | float | - | S-편광 진폭 |
| `s_phase` | float | radians | S-편광 위상 |
| `order` | int | - | 회절 차수 인덱스 (보통 0) |
| `direction` | int | - | 0: 위에서, 1: 아래에서 |

**편광 정의:**

**P-편광 (TM)**: 입사면 내의 E-장

$$
\hat{p} = \frac{\mathbf{k}_\parallel \times \hat{z} \times \mathbf{k}}{|\mathbf{k}_\parallel \times \hat{z} \times \mathbf{k}|}
$$

**S-편광 (TE)**: 입사면에 수직인 E-장

$$
\hat{s} = \frac{\mathbf{k} \times \hat{z}}{|\mathbf{k} \times \hat{z}|}
$$

**일반적인 경우:**

```python
# P-편광
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

# S-편광
obj.MakeExcitationPlanewave(0, 0, 1, 0, 0)

# 45° 선형 편광
obj.MakeExcitationPlanewave(1, 0, 1, 0, 0)

# 좌원 편광 (LCP)
obj.MakeExcitationPlanewave(1, 0, 1, np.pi/2, 0)

# 우원 편광 (RCP)
obj.MakeExcitationPlanewave(1, 0, 1, -np.pi/2, 0)
```

## 솔루션 변수

### 반사 및 투과

```python
R, T = obj.RT_Solve(normalize=1, byorder=0)
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `normalize` | int | 0: 원시, 1: 정규화 |
| `byorder` | int | 0: 전체, 1: 차수별 |

**반환:**

- `normalize=1, byorder=0`: $(R, T)$ 스칼라, 전체 파워
- `normalize=1, byorder=1`: $(R_i, T_i)$ 길이 `nG`의 배열

**정규화:**

`normalize=1`일 때:

$$
R = \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{refl}})}{\text{Re}(k_{z,\text{inc}})} |r_{mn}|^2
$$

$$
T = \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{trans}}) \varepsilon_{\text{in}}}{\text{Re}(k_{z,\text{inc}}) \varepsilon_{\text{out}}} |t_{mn}|^2
$$

에너지 보존: 무손실 구조의 경우 $R + T = 1$.

### 장 진폭

```python
a_i, b_i = obj.GetAmplitudes(which_layer, z_offset)
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `which_layer` | int | 레이어 인덱스 (0 기반) |
| `z_offset` | float | 레이어 내 위치 |

**반환:**

- `a_i`: 전방 모드 진폭, 형상 `(2*nG,)`, $(E_x, E_y)$ 성분
- `b_i`: 후방 모드 진폭, 형상 `(2*nG,)`

형식: `[Ex_0, Ex_1, ..., Ex_nG, Ey_0, Ey_1, ..., Ey_nG]`

### Fourier 공간의 장

```python
[Ex_mn, Ey_mn, Ez_mn], [Hx_mn, Hy_mn, Hz_mn] = obj.Solve_FieldFourier(which_layer, z_offset)
```

**반환:** 6개 배열, 각 형상 `(nG,)`, 복소수 값

- 각 성분의 Fourier 계수
- 전체 장을 얻으려면: 위상 인자와 함께 모든 차수 합산

### 실공간의 장

```python
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(which_layer, z_offset, Nxy)
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `Nxy` | list[int, int] | 그리드 크기 `[Nx, Ny]` |

**반환:** 6개 배열, 각 형상 `(Nx, Ny)`, 복소수 값

- 실공간 그리드의 전기장 및 자기장 성분
- 강도: $I = |E_x|^2 + |E_y|^2 + |E_z|^2$

## 내부 변수

### 레이어 저장

| 속성 | 설명 |
|-----------|-------------|
| `obj.Layer_N` | 전체 레이어 수 |
| `obj.thickness_list` | 레이어 두께 리스트 |
| `obj.id_list` | 레이어 타입 식별자 |
| `obj.Uniform_ep_list` | 균일 레이어의 유전 상수 |
| `obj.GridLayer_N` | 그리드 기반 레이어 수 |
| `obj.GridLayer_Nxy_list` | 각 패턴 레이어의 그리드 크기 |

### 고유 모드 변수

| 속성 | 설명 |
|-----------|-------------|
| `obj.q_list` | 각 레이어의 고유값 $q$ |
| `obj.phi_list` | 각 레이어의 고유벡터 |
| `obj.kp_list` | 각 레이어의 $K_\perp$ 행렬 |

이들은 `Init_Setup()`에 의해 계산되고 내부적으로 풀이에 사용됩니다.

## 일반적인 조합

### 스펙트럼 스윕

```python
wavelengths = np.linspace(0.4, 0.8, 100)
R_spectrum = []

for wl in wavelengths:
    freq = 1.0 / wl
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... 레이어 추가, 풀기 ...
    R, T = obj.RT_Solve(normalize=1)
    R_spectrum.append(R)
```

### 각도 스윕

```python
angles = np.linspace(0, 80, 50) * np.pi/180
R_angle = []

for theta in angles:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi=0, verbose=0)
    # ... 레이어 추가, 풀기 ...
    R, T = obj.RT_Solve(normalize=1)
    R_angle.append(R)
```

### 편광 스윕

```python
pols = np.linspace(0, np.pi, 90)  # 편광 각도
R_pol = []

for pol in pols:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... 레이어 추가 ...
    obj.MakeExcitationPlanewave(np.cos(pol), 0, np.sin(pol), 0, 0)
    R, T = obj.RT_Solve(normalize=1)
    R_pol.append(R)
```

## Autograd 변수

`grcwa.set_backend('autograd')` 사용 시:

```python
import autograd.numpy as np
from autograd import grad

# Autograd 가능한 매개변수
epsilon_grid = np.array(...)  # autograd.numpy 사용 필수
freq = np.array(1.0)
theta = np.array(0.1)
thickness = np.array(0.5)

# 경사도 계산
def objective(eps):
    # ... 설정 및 풀기 ...
    R, T = obj.RT_Solve()
    return -R  # R 최대화

grad_obj = grad(objective)
gradient = grad_obj(epsilon_grid)
```

**Autograd 가능한 매개변수:**

- ✅ 그리드의 유전 값
- ✅ 주파수 `freq`
- ✅ 각도 `theta`, `phi`
- ✅ 레이어 두께
- ✅ 주기성 스케일링 `Pscale`
- ❌ 절단 차수 `nG`
- ❌ 그리드 크기 `Nx`, `Ny`
- ❌ 레이어 수

## 요약 표

| 변수 | 기호 | 타입 | 단위 | 설명 |
|----------|--------|------|-------|-------------|
| 주파수 | $\omega/(2\pi)$ | float | 1/length | 작동 주파수 |
| 파장 | $\lambda$ | float | length | $\lambda = 1/f$ |
| 극각 | $\theta$ | float | radians | z축으로부터의 입사각 |
| 방위각 | $\phi$ | float | radians | xy평면의 각도 |
| 격자 벡터 | $\mathbf{L}_1, \mathbf{L}_2$ | list | length | 실공간 주기성 |
| 역격자 벡터 | $\mathbf{K}_1, \mathbf{K}_2$ | array | 1/length | Fourier 공간 주기성 |
| 절단 차수 | $N_G$ | int | - | Fourier 고조파 수 |
| 레이어 두께 | $d$ | float | length | 각 레이어의 두께 |
| 유전 상수 | $\varepsilon$ | complex | - | 상대 유전율 |
| 반사 | $R$ | float | - | 반사 파워 비율 |
| 투과 | $T$ | float | - | 투과 파워 비율 |

## 다음 단계

- **[물리 단위](units.md)**: 단위 변환 예제
- **[문제 해결](troubleshooting.md)**: 일반적인 문제 및 수정
- **[FAQ](faq.md)**: 자주 묻는 질문
- **[API 레퍼런스](../api/core.md)**: 완전한 함수 문서
