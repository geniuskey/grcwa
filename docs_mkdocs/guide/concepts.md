# 기본 개념

이 가이드는 GRCWA를 효과적으로 사용하는 데 필요한 기본 개념을 설명합니다.

## RCWA 작업 흐름

모든 RCWA 시뮬레이션은 다음 단계를 따릅니다:

```python
import grcwa
import numpy as np

# 1. RCWA 객체 생성
obj = grcwa.obj(nG, L1, L2, freq, theta, phi)

# 2. 레이어 스택 정의
obj.Add_LayerUniform(thickness1, epsilon1)
obj.Add_LayerGrid(thickness2, Nx, Ny)
obj.Add_LayerUniform(thickness3, epsilon3)

# 3. 초기화
obj.Init_Setup()

# 4. 패턴 입력 (그리드 레이어의 경우)
obj.GridLayer_geteps(epsilon_grid)

# 5. 여기 정의
obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0)

# 6. 풀기
R, T = obj.RT_Solve(normalize=1)

# 7. 장 분석 (선택 사항)
[Ex,Ey,Ez], [Hx,Hy,Hz] = obj.Solve_FieldOnGrid(layer, z, [Nx,Ny])
```

## 좌표계

GRCWA는 오른손 직교 좌표계를 사용합니다:

- **x, y**: 면내 방향 (주기적)
- **z**: 면외 방향 (레이어 적층)
- 빛은 **+z** 방향으로 전파

```
     z ↑
       |
       |  ┌────── 레이어 N
       |  ├────── 레이어 2
       |  ├────── 레이어 1
       |  └────── 레이어 0 (입력)
       └────────→ x
      ↙
     y
```

## 주기성 및 격자 벡터

### 격자 정의

두 격자 벡터 $\mathbf{L}_1$과 $\mathbf{L}_2$가 2D 주기 단위 셀을 정의합니다:

$$
\mathbf{L}_1 = (L_{1x}, L_{1y}, 0)
$$

$$
\mathbf{L}_2 = (L_{2x}, L_{2y}, 0)
$$

구조는 다음마다 반복됩니다:

$$
\mathbf{r} + m\mathbf{L}_1 + n\mathbf{L}_2 \quad (m, n \in \mathbb{Z})
$$

### 일반적인 격자

**정사각 격자** (주기 $a$):
```python
L1 = [a, 0]
L2 = [0, a]
```

**직사각 격자** (주기 $a, b$):
```python
L1 = [a, 0]
L2 = [0, b]
```

**육각 격자** (주기 $a$):
```python
L1 = [a, 0]
L2 = [a/2, a*np.sqrt(3)/2]
```

### 단위 셀 면적

단위 셀의 면적은:

$$
A_{\text{cell}} = |\mathbf{L}_1 \times \mathbf{L}_2| = |L_{1x}L_{2y} - L_{1y}L_{2x}|
$$

## 레이어 유형

### 1. 균일 레이어

상수 $\varepsilon$을 갖는 균질 유전체:

```python
obj.Add_LayerUniform(thickness, epsilon)
```

**장점:**

- 매우 빠름 (해석적 해)
- 수치적으로 안정적
- 패턴 입력 불필요

**사용처:**

- 진공/공기 영역
- 고체 유전체 슬랩
- 기판 레이어
- 클래딩 레이어

### 2. 그리드 기반 패턴 레이어

직교 그리드의 임의 2D 패턴:

```python
obj.Add_LayerGrid(thickness, Nx, Ny)
# 나중에:
obj.GridLayer_geteps(epsilon_grid.flatten())
```

**장점:**

- 최대 유연성 (모든 패턴)
- 복잡한 형상 정의가 쉬움
- 수치 최적화 지원

**사용처:**

- 광결정
- 메타표면
- 임의 패턴
- 최적화 문제

**그리드 해상도:**

- 매끄러운 패턴: Nx, Ny ≈ 50-100
- 날카로운 특징: Nx, Ny ≈ 200-500
- 매우 미세한 디테일: Nx, Ny ≈ 500-1000

### 3. Fourier 급수 레이어

해석적 Fourier 계수로 정의된 패턴:

```python
obj.Add_LayerFourier(thickness, params)
```

**장점:**

- FFT 불필요
- 알려진 기하학에 정확 (원, 직사각형)

**단점:**

- 알려진 Fourier 급수를 갖는 형상으로 제한
- 실제로 거의 사용되지 않음

## 절단 차수

### 그것은 무엇인가?

절단 차수 $N_G$는 포함되는 Fourier 조화함수(회절 차수)의 개수를 결정합니다:

$$
\mathbf{E}(\mathbf{r}) = \sum_{m,n} \mathbf{E}_{mn}(z) e^{i\mathbf{k}_{mn} \cdot \mathbf{r}_\parallel}
$$

합은 $N_G$ 항으로 절단됩니다.

### $N_G$ 선택

**경험 법칙:**

| 구조 유형 | 권장 $N_G$ |
|----------------|-------------------|
| 균일 레이어 | 11-51 |
| 매끄러운 패턴 | 51-101 |
| 날카로운 특징 | 101-301 |
| 매우 미세한 디테일 | 301-501 |
| 고정확도 | 501-1001 |

**트레이드오프:**

- ✅ 더 큰 $N_G$ → 더 정확
- ❌ 더 큰 $N_G$ → 더 느린 계산
- ❌ 더 큰 $N_G$ → 더 많은 메모리

### 수렴 테스트

항상 수렴을 테스트하세요:

```python
nG_values = [51, 101, 201, 301]
for nG in nG_values:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... 설정 ...
    R, T = obj.RT_Solve(normalize=1)
    print(f"nG={obj.nG:4d}: R={R:.6f}, T={T:.6f}")
```

$N_G$를 증가시켜도 결과가 변하지 않으면 수렴된 것입니다.

## 입사파

### 각도

**극각** $\theta$: z축(표면 법선)으로부터의 각도

- $\theta = 0$: 수직 입사
- $0 < \theta < 90°$: 경사 입사

**방위각** $\phi$: x축으로부터 xy-평면의 각도

- $\mathbf{k}$의 xy-평면으로의 투영 방향 정의

**입사 파동 벡터:**

$$
\mathbf{k}_{\text{inc}} = \omega\sqrt{\varepsilon_{\text{in}}}(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)
$$

### 편광

**P-편광 (TM):**

- 입사면 내의 전기장
- 입사면에 수직인 자기장

**S-편광 (TE):**

- 입사면에 수직인 전기장
- 입사면 내의 자기장

**임의 편광:**

$$
\mathbf{E} = A_p e^{i\phi_p} \hat{p} + A_s e^{i\phi_s} \hat{s}
$$

**일반적인 경우:**

```python
# P-편광
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

# S-편광
obj.MakeExcitationPlanewave(0, 0, 1, 0, 0)

# 선형 45°
obj.MakeExcitationPlanewave(1, 0, 1, 0, 0)

# 좌원 편광
obj.MakeExcitationPlanewave(1, 0, 1, np.pi/2, 0)

# 우원 편광
obj.MakeExcitationPlanewave(1, 0, 1, -np.pi/2, 0)
```

## 회절 차수

### 그것들은 무엇인가?

주기성으로 인해 입사파는 이산 회절 차수에 결합됩니다:

$$
\mathbf{k}_{mn,\parallel} = \mathbf{k}_{\parallel,0} + m\mathbf{K}_1 + n\mathbf{K}_2
$$

각 $(m,n)$은 회절 차수입니다.

### 전파 vs. 소산

각 차수에 대해 계산:

$$
k_{z,mn} = \sqrt{\varepsilon\omega^2 - k_{x,mn}^2 - k_{y,mn}^2}
$$

- **전파**: $k_z$가 실수 → 원거리장으로 파워 운반
- **소산**: $k_z$가 허수 → 지수적으로 감쇠

**예:**

```python
# Init_Setup() 후
for i in range(obj.nG):
    kx = obj.kx[i]
    ky = obj.ky[i]
    kz_sq = obj.omega**2 - kx**2 - ky**2  # 진공에서
    if kz_sq > 0:
        print(f"차수 {obj.G[i]}: 전파, kz={np.sqrt(kz_sq):.4f}")
    else:
        print(f"차수 {obj.G[i]}: 소산, κ={np.sqrt(-kz_sq):.4f}")
```

## 반사 및 투과

### 총 파워

```python
R, T = obj.RT_Solve(normalize=1)
```

- $R$: 총 반사 파워 (모든 차수)
- $T$: 총 투과 파워 (모든 차수)

손실 없는 구조의 경우: $R + T = 1$

### 차수별

```python
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
```

- `Ri[i]`: 차수 $i$의 반사 파워
- `Ti[i]`: 차수 $i$의 투과 파워

**분석:**

```python
print(f"0차: R={Ri[0]:.4f}, T={Ti[0]:.4f}")
print(f"고차: R={sum(Ri[1:]):.4f}, T={sum(Ti[1:]):.4f}")

# 어떤 차수가 상당한 파워를 운반하나?
threshold = 1e-3
for i in range(obj.nG):
    if Ri[i] > threshold or Ti[i] > threshold:
        m, n = obj.G[i]
        print(f"차수 ({m:2d},{n:2d}): R={Ri[i]:.4f}, T={Ti[i]:.4f}")
```

## 장 분석

### Fourier 공간

Fourier 계수 구하기:

```python
[Ex_mn, Ey_mn, Ez_mn], [Hx_mn, Hy_mn, Hz_mn] = obj.Solve_FieldFourier(layer, z_offset)
```

각 배열의 길이는 `nG`, 복소수 값입니다.

### 실공간

그리드에서 장 구하기:

```python
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(layer, z_offset, [Nx, Ny])
```

각 배열의 형상은 `(Nx, Ny)`, 복소수 값입니다.

**강도:**

```python
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
```

**Poynting 벡터:**

```python
Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
```

## 정규화

### 왜 정규화하나?

원시 RCWA 출력은 진폭입니다. 물리적 파워를 얻으려면 다음으로 정규화해야 합니다:

- 입사 파워
- 매질 특성 (임피던스)
- 각도 (투영 면적)

### 정규화 vs. 비정규화

```python
# 정규화 (권장)
R, T = obj.RT_Solve(normalize=1)
# 무손실의 경우 R + T = 1

# 비정규화 (원시 진폭)
R_raw, T_raw = obj.RT_Solve(normalize=0)
# 수동 정규화 필요
```

**물리적으로 의미 있는 결과를 위해 항상 `normalize=1`을 사용하세요.**

## 일반적인 함정

### 1. 에너지가 보존되지 않음

**증상:** $R + T \neq 1$

**원인:**

- 불충분한 $N_G$ (절단 차수 증가)
- 수치 불안정성 (레이어 두께 또는 $N_G$ 감소)
- 매우 높은 대비 구조 (더 많은 그리드 점 사용)

### 2. 잘못된 단위

**증상:** 이상한 결과, 비물리적 값

**해결책:** 일관된 단위 확인:

```python
# 모두 μm 단위
wavelength = 1.5  # μm
freq = 1.0 / wavelength
L1 = [0.6, 0]  # μm
thickness = 0.3  # μm
```

### 3. 그리드 해상도가 너무 낮음

**증상:** 잘못된 패턴, 들쭉날쭉한 가장자리

**해결책:** `Nx, Ny` 증가:

```python
# 낮은 해상도 (나쁨)
obj.Add_LayerGrid(0.3, 50, 50)

# 높은 해상도 (좋음)
obj.Add_LayerGrid(0.3, 400, 400)
```

### 4. 패턴 좌표

**실수:** 정규화 및 물리적 좌표 혼동

**올바름:**

```python
# [0,1] × [0,1]에 정의된 패턴 (정규화)
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# 물리적 좌표: 격자 벡터로 곱하기
x_phys = X * L1[0] + Y * L2[0]
y_phys = X * L1[1] + Y * L2[1]
```

### 5. 특이 행렬

**증상:** 고유값 풀이에서 오류

**원인:**

- 손실 없는 완전 수직 입사
- 퇴화 기하학

**해결책:** 작은 손실 추가:

```python
# freq = 1.0 대신
Qabs = 1e8  # 매우 높은 Q
freq = 1.0 * (1 + 1j / (2*Qabs))
```

## 모범 사례

### ✅ 해야 할 일

- $N_G$를 증가시키며 수렴 테스트
- R, T에 `normalize=1` 사용
- 무손실 구조에 대해 $R + T = 1$ 확인
- 충분한 그리드 해상도 사용 ($N_x, N_y \geq 200$)
- 에너지 보존 확인

### ❌ 하지 말아야 할 일

- 너무 적은 조화함수 사용 (수렴되지 않은 결과 위험)
- 풀기 전에 `Init_Setup()` 호출 잊기
- 단위 혼합 (예: 길이는 μm, 파장은 nm)
- 극도로 두꺼운 레이어 사용 (수치 불안정성)
- 에너지 보존 오류 무시

## 요약

GRCWA의 핵심 개념:

1. **작업 흐름**: 생성 → 레이어 추가 → 초기화 → 패턴 입력 → 여기 → 풀기
2. **격자**: $\mathbf{L}_1, \mathbf{L}_2$로 정의
3. **절단**: $N_G$ 조화함수, 수렴 테스트
4. **레이어**: 균일 (빠름) 또는 그리드 기반 (유연함)
5. **여기**: 각도 $(\theta, \phi)$ 및 편광 $(p, s)$
6. **결과**: $R, T$ 총계 또는 차수별
7. **장**: Fourier 또는 실공간

## 다음 단계

- **[레이어 정의](layers.md)**: 상세한 레이어 관리
- **[여기 설정](excitation.md)**: 편광 및 각도
- **[결과 계산](results.md)**: R, T 및 장 해석
- **[튜토리얼](../tutorials/tutorial1.md)**: 실습 예제
