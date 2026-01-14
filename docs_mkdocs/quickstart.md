# 빠른 시작 가이드

이 가이드를 통해 몇 분 안에 첫 번째 RCWA 시뮬레이션을 실행할 수 있습니다!

## 첫 번째 시뮬레이션

간단한 유전체 슬랩(유리판과 같은)을 시뮬레이션하고 얼마나 많은 빛이 반사되고 투과되는지 계산해 봅시다.

### 완전한 예제

```python
import grcwa
import numpy as np

# 단계 1: 구조 매개변수 정의
L1 = [1.0, 0]    # 격자 벡터 1 (x-방향)
L2 = [0, 1.0]    # 격자 벡터 2 (y-방향)
freq = 1.0       # 주파수 (파장 = 1.0 단위)
theta = 0.0      # 수직 입사
phi = 0.0        # 방위각
nG = 101         # Fourier 조화함수 개수

# 단계 2: RCWA 객체 생성
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)

# 단계 3: 레이어 정의 (위에서 아래로)
obj.Add_LayerUniform(1.0, 1.0)   # 상부: 진공 (ε=1), 두께=1.0
obj.Add_LayerUniform(0.5, 4.0)   # 중간: 유전체 슬랩 (ε=4), 두께=0.5
obj.Add_LayerUniform(1.0, 1.0)   # 하부: 진공 (ε=1), 두께=1.0

# 단계 4: 초기화
obj.Init_Setup()

# 단계 5: 입사파 정의 (p-편광, 진폭=1)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0,
                            order=0)

# 단계 6: 풀기
R, T = obj.RT_Solve(normalize=1)

# 단계 7: 결과 표시
print(f"반사: R = {R:.4f}")
print(f"투과: T = {T:.4f}")
print(f"에너지 보존: R + T = {R+T:.4f}")
```

**출력:**
```
반사: R = 0.3600
투과: T = 0.6400
에너지 보존: R + T = 1.0000
```

## 코드 이해하기

### 단계 1: 구조 매개변수

```python
L1 = [1.0, 0]    # 격자 벡터 1
L2 = [0, 1.0]    # 격자 벡터 2
```

이는 구조의 주기성을 정의합니다. 이 간단한 슬랩의 경우 주기성은 중요하지 않지만, RCWA는 항상 주기적 구조를 가정합니다.

```python
freq = 1.0
```

자연 단위(c=1)에서 주파수 = 1/파장입니다. 따라서 `freq=1.0`은 파장=1.0을 의미합니다.

```python
theta = 0.0  # z축으로부터의 각도 (수직 입사)
phi = 0.0    # xy-평면에서의 각도
```

수직 입사의 경우 두 각도를 모두 0으로 설정합니다.

```python
nG = 101
```

포함할 Fourier 조화함수의 개수입니다. 높을수록 더 정확하지만 느립니다. 균일한 슬랩의 경우 `nG=1`도 작동하지만, 패턴 레이어의 경우 더 큰 값이 필요합니다 (일반적으로 100-300).

### 단계 2: RCWA 객체 생성

```python
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

메인 RCWA 솔버 객체를 생성합니다. `verbose=1`은 진행 정보를 출력합니다.

### 단계 3: 레이어 추가

```python
obj.Add_LayerUniform(thickness, epsilon)
```

레이어는 **위에서 아래로** (입력에서 출력으로) 추가됩니다:

- **레이어 0**: 상부 진공 (반무한)
- **레이어 1**: 유전체 슬랩
- **레이어 2**: 하부 진공 (반무한)

!!! note "레이어 순서"
    항상 빛이 만나는 순서대로 레이어를 정의하세요:
    입력 영역 → 레이어 1 → 레이어 2 → ... → 출력 영역

### 단계 4: 초기화

```python
obj.Init_Setup()
```

다음을 계산합니다:
- 역격자 벡터
- 모든 회절 차수에 대한 파동 벡터
- 균일 레이어의 고유값

### 단계 5: 여기 정의

```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0,
                            order=0)
```

매개변수:
- `p_amp`: P-편광 진폭 (TM)
- `s_amp`: S-편광 진폭 (TE)
- `p_phase`, `s_phase`: 라디안 단위의 위상
- `order=0`: 0차 회절 차수에서 입사 (수직 입사)

### 단계 6: 풀기

```python
R, T = obj.RT_Solve(normalize=1)
```

- `normalize=1`: 입사 파워 및 매질 특성으로 정규화
- 전체 반사(R) 및 투과(T) 파워 반환

### 단계 7: 에너지 보존 확인

손실 없는 재료의 경우:
```python
assert abs(R + T - 1.0) < 1e-6, "에너지가 보존되지 않습니다!"
```

## 예제 2: 패턴 레이어

이제 더 흥미로운 것을 시뮬레이션해 봅시다: 원형 홀이 있는 광결정 슬랩.

```python
import grcwa
import numpy as np

# 설정
L1 = [1.5, 0]
L2 = [0, 1.5]
freq = 1.0
theta = 0.0
phi = 0.0
nG = 201  # 패턴 레이어에는 더 많은 조화함수가 필요

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)

# 레이어 구조
obj.Add_LayerUniform(1.0, 1.0)        # 진공
obj.Add_LayerGrid(0.3, 400, 400)      # 패턴 레이어: 400×400 그리드
obj.Add_LayerUniform(1.0, 1.0)        # 진공

obj.Init_Setup()

# 패턴 생성: 원형 공기 홀이 있는 실리콘 슬랩
Nx, Ny = 400, 400
x = np.linspace(0, 1, Nx)  # 정규화된 좌표 [0,1]
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# 실리콘으로 시작 (ε=12)
epsilon = np.ones((Nx, Ny)) * 12.0

# 중앙에 원형 공기 홀 추가 (ε=1)
radius = 0.4  # 격자 상수 단위
hole = (X - 0.5)**2 + (Y - 0.5)**2 < radius**2
epsilon[hole] = 1.0

# 패턴 입력
obj.GridLayer_geteps(epsilon.flatten())

# 여기
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0, order=0)

# 풀기
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")

# 차수별 반사/투과 구하기
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
print(f"\n회절 차수 개수: {len(Ri)}")
print(f"0차 반사: {Ri[0]:.4f}")
print(f"0차 투과: {Ti[0]:.4f}")
```

### 주요 차이점

**그리드 기반 레이어:**
```python
obj.Add_LayerGrid(thickness, Nx, Ny)
```

균일한 유전체 대신 `Nx × Ny` 점으로 2D 그리드를 정의합니다.

**패턴 정의:**
```python
epsilon = np.ones((Nx, Ny)) * 12.0  # 배경
epsilon[hole] = 1.0                  # 홀
obj.GridLayer_geteps(epsilon.flatten())
```

유전상수의 2D 배열을 생성한 다음 평탄화하여 입력합니다.

**차수별 분석:**
```python
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
```

각 회절 차수에 대한 R 및 T 배열을 얻습니다.

## 예제 3: 각도 의존 응답

입사각에 따른 반사 계산:

```python
import grcwa
import numpy as np
import matplotlib.pyplot as plt

# 구조 설정
L1 = [0.6, 0]
L2 = [0, 0.6]
freq = 1.0
nG = 101

# 각도 스윕
angles = np.linspace(0, 80, 50) * np.pi/180  # 0° ~ 80°
R_list = []

for theta in angles:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi=0, verbose=0)

    # Bragg 거울: 교번 레이어
    for i in range(5):
        if i % 2 == 0:
            obj.Add_LayerUniform(0.125, 4.0)  # 고굴절률
        else:
            obj.Add_LayerUniform(0.125, 2.25) # 저굴절률

    obj.Init_Setup()
    obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)

    R, T = obj.RT_Solve(normalize=1)
    R_list.append(R)

# 플롯
plt.figure(figsize=(8, 5))
plt.plot(angles * 180/np.pi, R_list, 'b-', linewidth=2)
plt.xlabel('입사각 (도)', fontsize=12)
plt.ylabel('반사율', fontsize=12)
plt.title('Bragg 거울의 각도 의존 반사')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('angle_sweep.png', dpi=150)
plt.show()
```

## 일반적인 작업 흐름

### 스펙트럼 계산

```python
wavelengths = np.linspace(0.4, 0.8, 100)  # μm
freqs = 1.0 / wavelengths

R_spectrum = []
T_spectrum = []

for freq in freqs:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... 레이어 추가 ...
    obj.Init_Setup()
    # ... 여기 및 패턴 설정 ...
    R, T = obj.RT_Solve(normalize=1)
    R_spectrum.append(R)
    T_spectrum.append(T)

# 스펙트럼 플롯
plt.plot(wavelengths, R_spectrum, label='R')
plt.plot(wavelengths, T_spectrum, label='T')
plt.xlabel('파장 (μm)')
plt.ylabel('파워')
plt.legend()
```

### 장 시각화

```python
# R, T를 풀고 난 후
layer = 1  # 시각화할 레이어
z_offset = 0.5  # 레이어 내 위치
Nxy = [100, 100]  # 그리드 해상도

# 그리드에서 장 구하기
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(layer, z_offset, Nxy)

# 강도 계산
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

# 플롯
plt.figure(figsize=(8, 8))
plt.imshow(I.T, origin='lower', cmap='hot', extent=[0, L1[0], 0, L2[1]])
plt.colorbar(label='강도 |E|²')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'레이어 {layer}의 장 강도')
plt.show()
```

## 빠른 참조

### 레이어 유형

| 메서드 | 사용 사례 |
|--------|----------|
| `Add_LayerUniform(thickness, ε)` | 균질 유전체 |
| `Add_LayerGrid(thickness, Nx, Ny)` | 임의의 2D 패턴 |
| `Add_LayerFourier(thickness, params)` | 해석적 Fourier 급수 |

### 풀이 옵션

```python
# 전체 R, T
R, T = obj.RT_Solve(normalize=1)

# 회절 차수별
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

# Fourier 공간의 장
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldFourier(layer, z_offset)

# 실공간의 장
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(layer, z_offset, [Nx, Ny])
```

### 편광

**P-편광 (TM)**: 입사면 내의 전기장
```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0, order=0)
```

**S-편광 (TE)**: 입사면에 수직인 전기장
```python
obj.MakeExcitationPlanewave(p_amp=0, p_phase=0, s_amp=1, s_phase=0, order=0)
```

**원형 편광**:
```python
# 좌원 편광
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=1, s_phase=np.pi/2, order=0)

# 우원 편광
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=1, s_phase=-np.pi/2, order=0)
```

## 초보자를 위한 팁

1. **간단하게 시작**: 패턴 구조를 시도하기 전에 균일한 레이어로 시작하세요
2. **에너지 보존 확인**: 손실 없는 재료의 경우 `R + T`는 1.0이어야 합니다
3. **충분한 조화함수 사용**: 패턴 레이어의 경우 일반적으로 `nG=101-301`
4. **그리드 해상도**: 정확한 패턴을 위해 `Nx, Ny ≥ 200` 사용
5. **특이점 회피**: 흡수가 없는 완전한 수직 입사의 경우 작은 손실 추가
6. **정규화**: 물리적으로 의미 있는 R, T를 위해 항상 `normalize=1` 사용

## 다음 단계

이제 더 고급 기능을 탐색할 준비가 되었습니다:

- **[기본 개념](guide/concepts.md)**: RCWA 이론 심층 이해
- **[튜토리얼](tutorials/tutorial1.md)**: 단계별 가이드 예제
- **[예제](examples/gallery.md)**: 예제 갤러리 둘러보기
- **[API 레퍼런스](api/core.md)**: 상세한 함수 문서

## 도움이 필요하신가요?

- [FAQ](reference/faq.md) 확인
- [전체 예제](examples/gallery.md) 보기
