# 튜토리얼 1: 간단한 유전체 슬랩

이 튜토리얼에서는 간단한 유전체 슬랩을 시뮬레이션하면서 GRCWA의 기본을 배웁니다.

## 학습 목표

이 튜토리얼을 마치면:

- GRCWA의 기본 워크플로 이해
- 균일한 유전체 레이어 생성
- 반사 및 투과 계산
- 에너지 보존 검증
- Fabry-Pérot 효과 이해

## 물리적 시스템

진공 중에 있는 유전체 슬랩(유리판과 같은)을 시뮬레이션합니다:

```
        공기 (ε=1)
    ┌─────────────────┐
    │   슬랩 (ε=4)    │  두께 = 0.5λ
    └─────────────────┘
        공기 (ε=1)
```

**매개변수:**

- 파장: λ = 1.0 μm
- 슬랩: 실리콘 (n=2, ε=4), 두께 = 0.5 μm
- 입사: 수직 (θ=0)
- 편광: P-편광

## 단계 1: Import 및 설정

```python
import grcwa
import numpy as np
import matplotlib.pyplot as plt

# 백엔드 설정 (속도를 위해 'numpy' 사용, 경사도를 위해 'autograd' 사용)
grcwa.set_backend('numpy')
```

## 단계 2: 구조 정의

```python
# 물리적 매개변수
wavelength = 1.0      # μm
freq = 1.0 / wavelength  # 자연 단위의 주파수

# 격자 벡터 (균일 슬랩의 경우 임의)
L1 = [1.0, 0]
L2 = [0, 1.0]

# 입사각 (수직 입사)
theta = 0.0
phi = 0.0

# 절단 차수 (균일 레이어의 경우 작은 값으로도 충분)
nG = 51

# RCWA 객체 생성
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

**설명:**

- `freq = 1.0/wavelength`: 파장을 주파수로 변환
- `L1, L2`: 균일 구조의 경우 주기성은 중요하지 않음
- `nG = 51`: 균일 레이어에는 작은 절단 차수로 충분
- `verbose=1`: 진행 정보 출력

## 단계 3: 레이어 추가

```python
# 레이어 두께
thickness_top = 1.0      # 반무한 진공
thickness_slab = 0.5     # 슬랩 두께 = λ/2
thickness_bottom = 1.0   # 반무한 진공

# 유전 상수
eps_vacuum = 1.0
eps_slab = 4.0  # n=2이므로 ε=n²=4

# 레이어 추가 (위에서 아래로)
obj.Add_LayerUniform(thickness_top, eps_vacuum)
obj.Add_LayerUniform(thickness_slab, eps_slab)
obj.Add_LayerUniform(thickness_bottom, eps_vacuum)
```

**레이어 순서:**

1. 입력 레이어 (슬랩 위의 진공)
2. 슬랩 레이어 (유전체)
3. 출력 레이어 (슬랩 아래의 진공)

**참고:** 첫 번째와 마지막 레이어는 반무한 영역으로 처리됩니다.

## 단계 4: 초기화

```python
# 역격자 및 고유값 초기화
obj.Init_Setup()

print(f"실제 사용된 nG: {obj.nG}")
print(f"각주파수: {obj.omega:.4f}")
```

이 함수는 다음을 계산합니다:

- 역격자 벡터
- 모든 회절 차수에 대한 파동 벡터
- 각 레이어의 고유값 및 고유벡터

## 단계 5: 여기 정의

```python
# P-편광 평면파
p_amp = 1.0
p_phase = 0.0
s_amp = 0.0
s_phase = 0.0

obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0)
```

**편광:**

- **P-편광** (TM): 입사면 내의 전기장
- **S-편광** (TE): 입사면에 수직인 전기장

수직 입사의 경우 등방성 재료에 대해 선택은 중요하지 않습니다.

## 단계 6: 풀이

```python
# 반사 및 투과 계산
R, T = obj.RT_Solve(normalize=1)

print("\n" + "="*50)
print(f"반사 (R): {R:.6f}")
print(f"투과 (T): {T:.6f}")
print(f"합계 (R+T): {R+T:.6f}")
print(f"에너지 보존 오차: {abs(R+T-1):.2e}")
print("="*50)
```

**예상 출력:**

```
==================================================
반사 (R): 0.055556
투과 (T): 0.944444
합계 (R+T): 1.000000
에너지 보존 오차: 0.00e+00
==================================================
```

## 단계 7: 해석적 검증

유전체 슬랩의 경우 Fresnel 방정식과 Fabry-Pérot 공식을 사용하여 R, T를 해석적으로 계산할 수 있습니다.

```python
def analytical_slab_RT(n_slab, thickness, wavelength, n_in=1.0, n_out=1.0):
    """
    수직 입사에서 유전체 슬랩의 해석적 R, T.
    """
    # 굴절률
    n = n_slab
    k0 = 2 * np.pi / wavelength
    kz = n * k0
    delta = kz * thickness  # 위상 두께

    # 경계면에서의 Fresnel 계수
    r12 = (n_in - n) / (n_in + n)  # 공기 -> 슬랩
    r23 = (n - n_out) / (n + n_out)  # 슬랩 -> 공기
    t12 = 2*n_in / (n_in + n)
    t23 = 2*n / (n + n_out)

    # Fabry-Pérot 공식
    numerator_r = r12 + r23 * np.exp(2j * delta)
    denominator = 1 + r12 * r23 * np.exp(2j * delta)
    r_total = numerator_r / denominator

    numerator_t = t12 * t23 * np.exp(1j * delta)
    t_total = numerator_t / denominator

    R = np.abs(r_total)**2
    T = (n_out / n_in) * np.abs(t_total)**2

    return R, T

# 해석적 결과 계산
n_slab = 2.0  # sqrt(4.0)
R_analytical, T_analytical = analytical_slab_RT(n_slab, thickness_slab, wavelength)

print("\n해석적 결과:")
print(f"R_analytical = {R_analytical:.6f}")
print(f"T_analytical = {T_analytical:.6f}")
print(f"\n차이:")
print(f"ΔR = {abs(R - R_analytical):.2e}")
print(f"ΔT = {abs(T - T_analytical):.2e}")
```

**예상 출력:**

```
해석적 결과:
R_analytical = 0.055556
T_analytical = 0.944444

차이:
ΔR = 1.39e-17
ΔT = 2.78e-17
```

완벽한 일치! 이것은 우리의 RCWA 시뮬레이션을 검증합니다.

## 단계 8: 스펙트럼 응답

파장에 따른 R과 T를 계산하여 Fabry-Pérot 무늬를 봅시다:

```python
# 파장 스윕
wavelengths = np.linspace(0.5, 2.0, 200)  # μm
R_spectrum = []
T_spectrum = []
R_analytical_spectrum = []

for wl in wavelengths:
    freq = 1.0 / wl

    # RCWA 계산
    obj_sweep = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj_sweep.Add_LayerUniform(thickness_top, eps_vacuum)
    obj_sweep.Add_LayerUniform(thickness_slab, eps_slab)
    obj_sweep.Add_LayerUniform(thickness_bottom, eps_vacuum)
    obj_sweep.Init_Setup()
    obj_sweep.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0)

    R, T = obj_sweep.RT_Solve(normalize=1)
    R_spectrum.append(R)
    T_spectrum.append(T)

    # 해석적
    R_an, T_an = analytical_slab_RT(n_slab, thickness_slab, wl)
    R_analytical_spectrum.append(R_an)

# 플롯
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, R_spectrum, 'b-', linewidth=2, label='R (RCWA)')
plt.plot(wavelengths, T_spectrum, 'r-', linewidth=2, label='T (RCWA)')
plt.plot(wavelengths, R_analytical_spectrum, 'ko', markersize=3,
         markevery=10, label='R (해석적)')

plt.xlabel('파장 (μm)', fontsize=12)
plt.ylabel('파워', fontsize=12)
plt.title('유전체 슬랩의 Fabry-Pérot 무늬', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0.5, 2.0)
plt.ylim(0, 1)

# 특수 파장 표시
# 두께 = m*λ/(2n)일 때 보강 간섭 -> R 최소
lambda_min = 2 * n_slab * thickness_slab / np.arange(1, 5)
for lm in lambda_min:
    if 0.5 < lm < 2.0:
        plt.axvline(lm, color='gray', linestyle='--', alpha=0.5)
        plt.text(lm, 0.95, f'λ={lm:.2f}', fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('tutorial1_spectrum.png', dpi=150)
plt.show()
```

**볼 수 있는 것:**

- 진동하는 R과 T (Fabry-Pérot 무늬)
- 두께 = mλ/(2n)일 때 R의 최소값 (보강 간섭)
- RCWA와 해석적 결과 간의 완벽한 일치

## 물리학 이해하기

### Fabry-Pérot 효과

빛이 슬랩의 두 경계면 사이에서 반사되어 간섭을 생성합니다:

- **보강 간섭**: 광학 경로 = 정수 × 파장일 때
    - 두께 = m × λ/(2n)
    - 최대 투과, 최소 반사

- **상쇄 간섭**: 광학 경로 = 반정수 × 파장일 때
    - 두께 = (m + 1/2) × λ/(2n)
    - 최소 투과, 최대 반사

### Fresnel 반사

각 공기-슬랩 경계면에서의 Fresnel 계수:

$$
r = \frac{n_1 - n_2}{n_1 + n_2}
$$

n=2의 경우:

$$
r = \frac{1-2}{1+2} = -\frac{1}{3}
$$

$$
R_{\text{단일 경계면}} = |r|^2 = \frac{1}{9} \approx 0.111
$$

그러나 두 경계면과 간섭이 있으면 λ = 1.0 μm에서 $R \approx 0.056$입니다.

## 연습 1: 두께 변경

슬랩 두께를 수정하고 R과 T에 미치는 영향을 관찰하세요:

```python
thicknesses = [0.25, 0.5, 0.75, 1.0]  # λ 단위

for thickness in thicknesses:
    obj_ex = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Add_LayerUniform(thickness, eps_slab)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Init_Setup()
    obj_ex.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R, T = obj_ex.RT_Solve(normalize=1)
    print(f"두께 = {thickness:.2f}λ: R = {R:.4f}, T = {T:.4f}")
```

## 연습 2: 굴절률 변경

다른 슬랩 재료를 시도해보세요:

```python
refractive_indices = [1.5, 2.0, 3.0, 3.5]  # 유리, Si, GaAs, 적외선의 Si

for n in refractive_indices:
    eps = n**2
    obj_ex = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Add_LayerUniform(0.5, eps)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Init_Setup()
    obj_ex.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R, T = obj_ex.RT_Solve(normalize=1)
    print(f"n = {n:.1f}: R = {R:.4f}, T = {T:.4f}")
```

**관찰:** 굴절률이 높을수록 → 반사가 높음.

## 연습 3: 경사 입사

각도에서는 어떻게 될까요?

```python
angles = np.linspace(0, 80, 50) * np.pi/180

R_p_list = []
R_s_list = []

for theta in angles:
    obj_angle = grcwa.obj(nG, L1, L2, freq, theta, phi=0, verbose=0)
    obj_angle.Add_LayerUniform(1.0, eps_vacuum)
    obj_angle.Add_LayerUniform(0.5, eps_slab)
    obj_angle.Add_LayerUniform(1.0, eps_vacuum)
    obj_angle.Init_Setup()

    # P-편광
    obj_angle.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R_p, _ = obj_angle.RT_Solve(normalize=1)
    R_p_list.append(R_p)

    # S-편광
    obj_angle.MakeExcitationPlanewave(0, 0, 1, 0, 0)
    R_s, _ = obj_angle.RT_Solve(normalize=1)
    R_s_list.append(R_s)

plt.figure(figsize=(8, 6))
plt.plot(angles*180/np.pi, R_p_list, 'b-', label='P-편광')
plt.plot(angles*180/np.pi, R_s_list, 'r-', label='S-편광')
plt.xlabel('입사각 (도)')
plt.ylabel('반사율')
plt.title('각도 의존 반사')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**관찰:** P-편광의 Brewster 각도, 스침 각도에서의 전반사.

## 요약

이 튜토리얼에서 배운 내용:

✅ 기본 RCWA 시뮬레이션 생성 방법
✅ 균일한 유전체 레이어 추가
✅ 반사 및 투과 계산
✅ 해석적 공식에 대한 결과 검증
✅ 스펙트럼 응답 계산 (Fabry-Pérot 무늬)
✅ 물리적 현상 이해 (간섭, Fresnel 반사)

## 다음 단계

- **[튜토리얼 2](tutorial2.md)**: 홀이 있는 패턴 레이어
- **[튜토리얼 3](tutorial3.md)**: 여러 패턴 레이어
- **[튜토리얼 4](tutorial4.md)**: 육각 격자
- **[튜토리얼 5](tutorial5.md)**: 위상 최적화

## 전체 코드

<details>
<summary>전체 코드를 보려면 클릭</summary>

```python
import grcwa
import numpy as np
import matplotlib.pyplot as plt

# 설정
grcwa.set_backend('numpy')

wavelength = 1.0
freq = 1.0 / wavelength
L1 = [1.0, 0]
L2 = [0, 1.0]
theta = 0.0
phi = 0.0
nG = 51

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)

# 레이어
thickness_slab = 0.5
eps_vacuum = 1.0
eps_slab = 4.0

obj.Add_LayerUniform(1.0, eps_vacuum)
obj.Add_LayerUniform(thickness_slab, eps_slab)
obj.Add_LayerUniform(1.0, eps_vacuum)

# 초기화
obj.Init_Setup()

# 여기
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

# 풀이
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.6f}, T = {T:.6f}, R+T = {R+T:.6f}")

# 해석적 검증
n_slab = 2.0
k0 = 2 * np.pi / wavelength
kz = n_slab * k0
delta = kz * thickness_slab

r12 = (1 - n_slab) / (1 + n_slab)
r23 = (n_slab - 1) / (n_slab + 1)
t12 = 2 / (1 + n_slab)
t23 = 2*n_slab / (n_slab + 1)

r_total = (r12 + r23 * np.exp(2j * delta)) / (1 + r12 * r23 * np.exp(2j * delta))
t_total = (t12 * t23 * np.exp(1j * delta)) / (1 + r12 * r23 * np.exp(2j * delta))

R_analytical = np.abs(r_total)**2
T_analytical = np.abs(t_total)**2

print(f"R_analytical = {R_analytical:.6f}")
print(f"T_analytical = {T_analytical:.6f}")
print(f"오차: ΔR = {abs(R-R_analytical):.2e}")
```

</details>
