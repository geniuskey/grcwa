# 이미지 센서 픽셀 배열 시뮬레이션 가이드

이 가이드는 CMOS/CCD 이미지 센서의 픽셀 배열을 GRCWA로 시뮬레이션할 때 정합성을 높이는 방법을 설명합니다.

## 이미지 센서 구조 이해

### 전형적인 픽셀 스택

```
       입사광 ↓
┌─────────────────────┐
│  마이크로렌즈       │ ← 광 집광
├─────────────────────┤
│  컬러 필터 (RGB)    │ ← 파장 선택
├─────────────────────┤
│  금속 배선 층       │ ← 차광/배선
├─────────────────────┤
│  포토다이오드       │ ← 광전 변환
└─────────────────────┘
      실리콘 기판
```

### 주요 구조적 특징

- **픽셀 피치**: 0.8-10 μm (최신 센서: 0.7-1.4 μm)
- **마이크로렌즈**: 곡률 반경 ~ 픽셀 피치
- **컬러 필터**: 두께 0.5-1.0 μm, Bayer/RGGB/RGBW 패턴
- **금속 배선**: 여러 층, 개구율 영향
- **포토다이오드**: 깊이 1-3 μm

!!! warning "RCWA 적용 시 고려사항"
    - RCWA는 **주기 구조**를 가정합니다
    - 실제 센서는 유한 크기이지만 주기 경계 조건으로 근사
    - 가장자리 효과는 무시됩니다

## 단위 셀 선택 전략

### 1. Bayer 패턴의 경우

가장 일반적인 컬러 필터 배열:

```
┌──────┬──────┐
│  G   │  R   │
├──────┼──────┤
│  B   │  G   │
└──────┴──────┘
```

**단위 셀 선택:**

=== "최소 단위 셀 (권장)"
    ```python
    # 2×2 Bayer 패턴 = 최소 반복 단위
    pixel_pitch = 1.12  # μm
    L1 = [2 * pixel_pitch, 0]
    L2 = [0, 2 * pixel_pitch]

    obj = grcwa.obj(nG=201, L1=L1, L2=L2, freq, theta, phi)
    ```

    **장점**: 계산 효율적, 메모리 절약

    **단점**: 대칭성이 낮음

=== "4×4 슈퍼셀"
    ```python
    # 더 큰 슈퍼셀로 통계적 변동 포함
    L1 = [4 * pixel_pitch, 0]
    L2 = [0, 4 * pixel_pitch]

    obj = grcwa.obj(nG=301, L1=L1, L2=L2, freq, theta, phi)
    ```

    **장점**: 제조 변동, 불균일성 포함 가능

    **단점**: 계산 비용 4배 증가

### 2. 단색 센서의 경우

마이크로렌즈만 있는 경우:

```python
# 단일 픽셀 = 단위 셀
L1 = [pixel_pitch, 0]
L2 = [0, pixel_pitch]

obj = grcwa.obj(nG=151, L1=L1, L2=L2, freq, theta, phi)
```

### 3. 편광 센서의 경우

편광 필터 배열 (예: Sony Polarsens):

```
┌────┬────┐
│ 0° │ 45°│
├────┼────┤
│135°│ 90°│
└────┴────┘
```

2×2 단위 셀 사용, 각 필터의 편광 방향 정확히 모델링 필요.

## 레이어 구성 및 모델링

### 레이어 순서 (위에서 아래)

```python
import grcwa
import numpy as np

# 파라미터
wavelength = 0.55  # μm (녹색광)
freq = 1.0 / wavelength
pixel_pitch = 1.12  # μm
nG = 201

# 단위 셀 정의
L1 = [2*pixel_pitch, 0]  # Bayer 2×2
L2 = [0, 2*pixel_pitch]

obj = grcwa.obj(nG, L1, L2, freq, theta=0, phi=0, verbose=1)

# 레이어 1: 상부 공기
obj.Add_LayerUniform(thickness=1.0, epsilon=1.0)

# 레이어 2: 마이크로렌즈 (패턴)
obj.Add_LayerGrid(thickness=0.8, Nx=400, Ny=400)

# 레이어 3: 컬러 필터 (패턴)
obj.Add_LayerGrid(thickness=0.6, Nx=400, Ny=400)

# 레이어 4: 금속 배선 층 (패턴)
obj.Add_LayerGrid(thickness=0.3, Nx=400, Ny=400)

# 레이어 5: 실리콘 기판
eps_Si = (3.88 + 0.02j)**2  # 가시광 영역
obj.Add_LayerUniform(thickness=2.0, epsilon=eps_Si)

# 초기화
obj.Init_Setup()
```

## 정합성 향상 기법

### 1. 마이크로렌즈 모델링

정확한 3D 프로파일 필요:

```python
def create_microlens_profile(Nx, Ny, lens_params):
    """
    정확한 마이크로렌즈 프로파일 생성

    Parameters:
    -----------
    Nx, Ny : int
        그리드 해상도
    lens_params : dict
        radius: 렌즈 곡률 반경 (μm)
        sag: 렌즈 높이 (μm)
        pitch: 픽셀 피치 (μm)
        n_lens: 렌즈 굴절률 (일반적으로 1.5-1.6)
    """
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    pitch = lens_params['pitch']
    radius = lens_params['radius']
    n_lens = lens_params['n_lens']

    # Bayer 2×2 패턴의 경우 4개 렌즈
    eps_array = np.ones((Nx, Ny))  # 공기

    # 각 픽셀 위치에 렌즈 배치
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        center_x = (i + 0.5) * 0.5
        center_y = (j + 0.5) * 0.5

        # 렌즈 영역 정의 (구면 근사)
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        lens_mask = dist < (0.45 * 0.5)  # 픽셀의 90% 면적

        eps_array[lens_mask] = n_lens**2

    return eps_array

# 사용
lens_params = {
    'pitch': 2.24,  # 2×2 단위 셀
    'radius': 1.5,
    'sag': 0.3,
    'n_lens': 1.55
}

eps_lens = create_microlens_profile(400, 400, lens_params)
```

!!! tip "고급 기법"
    더 정확한 모델링을 위해 렌즈를 **여러 층으로 분할**:
    ```python
    # 렌즈를 5개 얇은 층으로 분할
    for i in range(5):
        z = i / 5.0
        thickness_i = 0.16
        eps_i = get_lens_profile_at_z(z)  # z에서의 단면
        obj.Add_LayerGrid(thickness_i, 400, 400)
    ```

### 2. 컬러 필터 모델링

실제 염료 특성 반영:

```python
def create_color_filter_bayer(Nx, Ny, filter_params):
    """
    Bayer 컬러 필터 패턴 생성

    Parameters:
    -----------
    filter_params : dict
        n_R, n_G, n_B: RGB 필터의 복소 굴절률
        transmission: 각 색상의 투과율 (선택적)
    """
    eps_array = np.ones((Nx, Ny), dtype=complex)

    # 2×2 Bayer 패턴 정의
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # G-R / B-G 배열
    mask_G1 = (X < 0.5) & (Y < 0.5)  # 좌상
    mask_R = (X >= 0.5) & (Y < 0.5)   # 우상
    mask_B = (X < 0.5) & (Y >= 0.5)   # 좌하
    mask_G2 = (X >= 0.5) & (Y >= 0.5) # 우하

    n_G = filter_params['n_G']
    n_R = filter_params['n_R']
    n_B = filter_params['n_B']

    eps_array[mask_G1 | mask_G2] = n_G**2
    eps_array[mask_R] = n_R**2
    eps_array[mask_B] = n_B**2

    return eps_array

# 파장 의존 굴절률 (Cauchy 모델)
def get_filter_index(wavelength, color):
    """파장에 따른 컬러 필터 굴절률"""
    # 실제 측정 데이터 또는 제조사 스펙 사용
    if color == 'R':
        n = 1.50 + 0.01/(wavelength**2)  # 예시
        k = 0.1 if wavelength > 0.6 else 1.5  # 흡수
    elif color == 'G':
        n = 1.52 + 0.01/(wavelength**2)
        k = 0.1 if 0.5 < wavelength < 0.6 else 1.2
    elif color == 'B':
        n = 1.48 + 0.01/(wavelength**2)
        k = 0.1 if wavelength < 0.5 else 1.8
    return n + 1j*k

# 파장별 시뮬레이션
wavelengths = np.linspace(0.4, 0.7, 30)
QE_spectrum = {'R': [], 'G': [], 'B': []}

for wl in wavelengths:
    freq = 1.0 / wl

    # 파장에 따른 필터 특성 업데이트
    filter_params = {
        'n_R': get_filter_index(wl, 'R'),
        'n_G': get_filter_index(wl, 'G'),
        'n_B': get_filter_index(wl, 'B')
    }

    eps_filter = create_color_filter_bayer(400, 400, filter_params)

    # ... 시뮬레이션 수행 ...
```

### 3. 금속 배선 층 모델링

실제 레이아웃 반영:

```python
def create_metal_layer(Nx, Ny, metal_params):
    """
    금속 배선 구조 생성

    Parameters:
    -----------
    metal_params : dict
        aperture_ratio: 개구율 (0-1)
        eps_metal: 금속 유전율 (Al, Cu 등)
        eps_dielectric: 절연체 유전율
    """
    eps_array = np.ones((Nx, Ny), dtype=complex)

    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    aperture = metal_params['aperture_ratio']
    eps_metal = metal_params['eps_metal']
    eps_diel = metal_params['eps_dielectric']

    # 기본적으로 절연체
    eps_array[:] = eps_diel

    # 각 픽셀의 중앙에 개구부 (광 통과)
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        center_x = (i + 0.5) * 0.5
        center_y = (j + 0.5) * 0.5

        # 개구부 크기
        aperture_size = np.sqrt(aperture) * 0.5

        in_aperture = (np.abs(X - center_x) < aperture_size/2) & \
                     (np.abs(Y - center_y) < aperture_size/2)

        # 개구부는 절연체 유지, 나머지는 금속
        eps_array[~in_aperture &
                 (np.abs(X - center_x) < 0.45) &
                 (np.abs(Y - center_y) < 0.45)] = eps_metal

    return eps_array

# 알루미늄 배선 예시 (Drude 모델)
def eps_aluminum(wavelength):
    """알루미늄 유전율 (Drude 모델)"""
    omega = 2*np.pi / wavelength
    omega_p = 15.0  # eV
    gamma = 0.1     # eV
    eps_inf = 1.0

    eps = eps_inf - omega_p**2 / (omega**2 + 1j*omega*gamma)
    return eps

metal_params = {
    'aperture_ratio': 0.64,  # 80% × 80%
    'eps_metal': eps_aluminum(0.55),
    'eps_dielectric': (1.46)**2  # SiO2
}

eps_metal = create_metal_layer(400, 400, metal_params)
```

## 그리드 해상도 최적화

### 수렴 테스트

```python
def test_grid_convergence(Nx_values, structure_params):
    """
    그리드 해상도 수렴 테스트

    Returns:
    --------
    dict: {Nx: (R, T, computation_time)}
    """
    import time
    results = {}

    for Nx in Nx_values:
        Ny = Nx

        start = time.time()

        # 시뮬레이션 설정
        obj = grcwa.obj(nG=201, L1=L1, L2=L2, freq, theta=0, phi=0, verbose=0)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(0.8, Nx, Ny)  # 마이크로렌즈
        obj.Add_LayerGrid(0.6, Nx, Ny)  # 컬러 필터
        obj.Add_LayerGrid(0.3, Nx, Ny)  # 금속 층
        obj.Add_LayerUniform(2.0, eps_Si)

        obj.Init_Setup()

        # 패턴 생성 및 입력
        eps_lens = create_microlens_profile(Nx, Ny, lens_params)
        eps_filter = create_color_filter_bayer(Nx, Ny, filter_params)
        eps_metal = create_metal_layer(Nx, Ny, metal_params)

        eps_all = np.concatenate([
            eps_lens.flatten(),
            eps_filter.flatten(),
            eps_metal.flatten()
        ])
        obj.GridLayer_geteps(eps_all)

        # 여기 및 계산
        obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
        R, T = obj.RT_Solve(normalize=1)

        elapsed = time.time() - start
        results[Nx] = (R, T, elapsed)

        print(f"Nx={Nx}: R={R:.4f}, T={T:.4f}, time={elapsed:.1f}s")

    return results

# 테스트 실행
Nx_values = [100, 200, 300, 400, 500]
results = test_grid_convergence(Nx_values, structure_params)

# 수렴 플롯
import matplotlib.pyplot as plt

Nx_list = list(results.keys())
R_list = [results[Nx][0] for Nx in Nx_list]
T_list = [results[Nx][1] for Nx in Nx_list]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(Nx_list, R_list, 'o-', label='Reflection')
plt.plot(Nx_list, T_list, 's-', label='Transmission')
plt.xlabel('Grid Resolution (Nx)')
plt.ylabel('Power')
plt.legend()
plt.title('Grid Convergence')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
times = [results[Nx][2] for Nx in Nx_list]
plt.plot(Nx_list, times, 'd-')
plt.xlabel('Grid Resolution (Nx)')
plt.ylabel('Computation Time (s)')
plt.title('Computational Cost')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grid_convergence.png', dpi=150)
```

!!! success "권장 해상도"
    **일반적인 가이드라인:**

    - **마이크로렌즈**: Nx, Ny ≥ 200 (곡면 표현)
    - **컬러 필터**: Nx, Ny ≥ 100 (경계 선명)
    - **금속 층**: Nx, Ny ≥ 150 (개구부 모서리)

    **트레이드오프**: Nx = 400은 정확도와 속도의 좋은 균형

## 절단 차수 최적화

### nG 수렴 테스트

```python
def test_nG_convergence(nG_values):
    """절단 차수 수렴 테스트"""
    results = {}

    for nG in nG_values:
        obj = grcwa.obj(nG, L1, L2, freq, theta=0, phi=0, verbose=0)
        # ... 레이어 추가 및 시뮬레이션 ...
        R, T = obj.RT_Solve(normalize=1)
        results[nG] = (R, T)

        print(f"nG={nG}: actual_nG={obj.nG}, R={R:.6f}, T={T:.6f}")

    return results

nG_values = [51, 101, 151, 201, 301, 401]
nG_results = test_nG_convergence(nG_values)
```

!!! tip "nG 선택 기준"
    **픽셀 피치에 따른 권장값:**

    | 픽셀 피치 (μm) | 파장 (μm) | 추천 nG |
    |---------------|----------|---------|
    | > 3.0         | 0.4-0.7  | 101-151 |
    | 1.5-3.0       | 0.4-0.7  | 151-201 |
    | 1.0-1.5       | 0.4-0.7  | 201-301 |
    | < 1.0         | 0.4-0.7  | 301-501 |

    **기준**: 픽셀 피치 / 파장 비율이 작을수록 더 많은 회절 차수 필요

## 양자 효율(QE) 계산

### 포토다이오드 흡수 계산

```python
def calculate_quantum_efficiency(obj, which_layer_PD, wavelength):
    """
    포토다이오드 층의 양자 효율 계산

    Parameters:
    -----------
    obj : grcwa.obj
        시뮬레이션 객체
    which_layer_PD : int
        포토다이오드 레이어 인덱스
    wavelength : float
        파장 (μm)

    Returns:
    --------
    QE : float
        양자 효율 (0-1)
    """
    # 포토다이오드 층에서 흡수된 파워 계산
    # 방법 1: 장 적분 사용
    thickness_PD = obj.thickness_list[which_layer_PD]

    # 층 내 여러 위치에서 장 샘플링
    z_positions = np.linspace(0, thickness_PD, 10)
    absorbed_power = 0

    for z in z_positions:
        # 실공간 장 계산
        [Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(
            which_layer_PD, z, [100, 100]
        )

        # 장 강도
        E_intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

        # 실리콘 흡수 계수 (파장 의존)
        alpha = get_Si_absorption(wavelength)  # cm^-1
        alpha_um = alpha * 1e-4  # μm^-1

        # 국소 흡수
        local_absorption = alpha_um * np.mean(E_intensity)
        absorbed_power += local_absorption

    # 평균 흡수
    absorbed_power /= len(z_positions)

    # 입사 파워로 정규화
    R, T = obj.RT_Solve(normalize=1)
    incident_power = 1.0

    # QE = 흡수된 파워 / 입사 파워
    QE = absorbed_power / incident_power

    return min(QE, 1.0)  # 최대 100%

def get_Si_absorption(wavelength):
    """
    실리콘 흡수 계수 (실험 데이터)

    Parameters:
    -----------
    wavelength : float (μm)

    Returns:
    --------
    alpha : float (cm^-1)
    """
    # 실제 측정 데이터 (Green & Keevers, 1995)
    data = {
        0.4: 1.0e5,
        0.5: 1.0e4,
        0.55: 5.0e3,
        0.6: 3.0e3,
        0.7: 1.0e3,
        0.8: 300,
        0.9: 100,
        1.0: 30
    }

    # 선형 보간
    wl_array = np.array(list(data.keys()))
    alpha_array = np.array(list(data.values()))
    alpha = np.interp(wavelength, wl_array, alpha_array)

    return alpha
```

### 픽셀별 QE 분석

```python
def analyze_pixel_QE(obj, wavelengths, pixel_types=['R', 'G', 'B']):
    """
    Bayer 패턴의 각 픽셀 타입별 QE 계산

    Returns:
    --------
    dict: {pixel_type: QE_spectrum}
    """
    QE_results = {ptype: [] for ptype in pixel_types}

    for wl in wavelengths:
        freq = 1.0 / wl

        # 시뮬레이션 설정 (파장 의존 재질 특성 업데이트)
        obj = setup_sensor_simulation(wl)

        # 포토다이오드 층에서 장 계산
        [Ex, Ey, Ez], _ = obj.Solve_FieldOnGrid(
            which_layer=4,  # PD 층
            z_offset=1.0,   # 중간 깊이
            Nxy=[200, 200]
        )

        # 강도 맵
        intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

        # 각 픽셀 타입별로 평균 강도 계산
        # Bayer: G(0,0), R(1,0), B(0,1), G(1,1)
        Nx, Ny = intensity.shape

        # G 픽셀 (2개)
        G1_region = intensity[:Nx//2, :Ny//2]
        G2_region = intensity[Nx//2:, Ny//2:]
        QE_G = (np.mean(G1_region) + np.mean(G2_region)) / 2

        # R 픽셀
        R_region = intensity[Nx//2:, :Ny//2]
        QE_R = np.mean(R_region)

        # B 픽셀
        B_region = intensity[:Nx//2, Ny//2:]
        QE_B = np.mean(B_region)

        # 흡수 계수 적용
        alpha = get_Si_absorption(wl)
        thickness_PD = 2.0  # μm

        QE_results['R'].append(QE_R * alpha * thickness_PD * 1e-4)
        QE_results['G'].append(QE_G * alpha * thickness_PD * 1e-4)
        QE_results['B'].append(QE_B * alpha * thickness_PD * 1e-4)

    return QE_results

# QE 스펙트럼 계산
wavelengths = np.linspace(0.4, 0.9, 50)
QE_spectrum = analyze_pixel_QE(obj, wavelengths)

# 플롯
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, QE_spectrum['R'], 'r-', linewidth=2, label='R pixel')
plt.plot(wavelengths, QE_spectrum['G'], 'g-', linewidth=2, label='G pixel')
plt.plot(wavelengths, QE_spectrum['B'], 'b-', linewidth=2, label='B pixel')
plt.xlabel('Wavelength (μm)', fontsize=12)
plt.ylabel('Quantum Efficiency', fontsize=12)
plt.title('Spectral QE of Bayer Pixels', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.savefig('QE_spectrum.png', dpi=150)
```

## 검증 및 보정

### 1. 에너지 보존 확인

```python
def verify_energy_conservation(obj):
    """에너지 보존 검증"""
    R, T = obj.RT_Solve(normalize=1)

    # 포토다이오드 흡수 계산
    A = 1 - R - T  # 흡수

    print(f"Reflection:    {R:.4f}")
    print(f"Transmission:  {T:.4f}")
    print(f"Absorption:    {A:.4f}")
    print(f"Sum (R+T+A):   {R+T+A:.4f}")
    print(f"Error:         {abs(R+T+A-1):.2e}")

    if abs(R+T+A-1) > 0.01:
        print("⚠️  WARNING: Energy not conserved! Check:")
        print("   - Loss in metal layers")
        print("   - Absorption in dielectrics")
        print("   - Numerical convergence")
    else:
        print("✓ Energy conservation satisfied")

    return R, T, A

verify_energy_conservation(obj)
```

### 2. 실측 데이터와 비교

#### 2.1 실측 데이터 형식 및 로딩

실측 QE 데이터는 보통 다음 형식 중 하나입니다:

**CSV 형식 예시:**
```csv
# sensor_QE_data.csv
Wavelength_nm,R_QE,G_QE,B_QE,R_std,G_std,B_std
400,0.12,0.15,0.45,0.01,0.01,0.02
450,0.25,0.35,0.65,0.02,0.02,0.03
500,0.45,0.55,0.35,0.02,0.03,0.02
...
```

**엑셀 형식:**
- Sheet 1: R pixel QE vs wavelength
- Sheet 2: G pixel QE vs wavelength
- Sheet 3: B pixel QE vs wavelength

```python
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit

def load_measured_QE_from_csv(filename, wavelength_col='Wavelength_nm'):
    """
    CSV 파일에서 실측 QE 데이터 로딩

    Parameters:
    -----------
    filename : str
        CSV 파일 경로
    wavelength_col : str
        파장 컬럼 이름

    Returns:
    --------
    data : dict
        {'wavelengths': array, 'R': array, 'G': array, 'B': array,
         'R_std': array, 'G_std': array, 'B_std': array}
    """
    df = pd.read_csv(filename, comment='#')

    # 파장을 μm 단위로 변환 (nm → μm)
    wavelengths = df[wavelength_col].values / 1000.0

    # QE 데이터
    data = {
        'wavelengths': wavelengths,
        'R': df['R_QE'].values if 'R_QE' in df.columns else None,
        'G': df['G_QE'].values if 'G_QE' in df.columns else None,
        'B': df['B_QE'].values if 'B_QE' in df.columns else None,
    }

    # 표준편차 (있는 경우)
    if 'R_std' in df.columns:
        data['R_std'] = df['R_std'].values
        data['G_std'] = df['G_std'].values
        data['B_std'] = df['B_std'].values
    else:
        # 표준편차가 없으면 5% 추정
        data['R_std'] = data['R'] * 0.05 if data['R'] is not None else None
        data['G_std'] = data['G'] * 0.05 if data['G'] is not None else None
        data['B_std'] = data['B'] * 0.05 if data['B'] is not None else None

    return data

def load_measured_QE_from_excel(filename):
    """
    Excel 파일에서 실측 QE 데이터 로딩 (시트별로 R/G/B)
    """
    data = {'wavelengths': None}

    for color in ['R', 'G', 'B']:
        df = pd.read_excel(filename, sheet_name=color)
        if data['wavelengths'] is None:
            data['wavelengths'] = df['Wavelength_nm'].values / 1000.0
        data[color] = df['QE'].values
        data[f'{color}_std'] = df['QE_std'].values if 'QE_std' in df.columns else data[color] * 0.05

    return data

# 실측 데이터 로드
measured_data = load_measured_QE_from_csv('sensor_QE_measurement.csv')
print(f"Loaded {len(measured_data['wavelengths'])} wavelength points")
print(f"Wavelength range: {measured_data['wavelengths'][0]:.3f} - {measured_data['wavelengths'][-1]:.3f} μm")
```

#### 2.2 데이터 전처리 및 정렬

실측과 시뮬레이션 데이터의 파장 샘플링이 다를 수 있습니다:

```python
def align_wavelengths(sim_wavelengths, sim_QE, meas_wavelengths, meas_QE, meas_std=None):
    """
    시뮬레이션과 실측 데이터의 파장 정렬

    Parameters:
    -----------
    sim_wavelengths : array
        시뮬레이션 파장 배열
    sim_QE : array
        시뮬레이션 QE
    meas_wavelengths : array
        실측 파장 배열
    meas_QE : array
        실측 QE
    meas_std : array, optional
        실측 표준편차

    Returns:
    --------
    aligned_data : dict
        정렬된 데이터 {'wavelengths', 'sim_QE', 'meas_QE', 'meas_std'}
    """
    # 공통 파장 범위 찾기
    wl_min = max(sim_wavelengths[0], meas_wavelengths[0])
    wl_max = min(sim_wavelengths[-1], meas_wavelengths[-1])

    # 공통 범위에서 새 파장 배열 생성
    common_wavelengths = np.linspace(wl_min, wl_max, 50)

    # 보간
    sim_interp = interpolate.interp1d(sim_wavelengths, sim_QE, kind='cubic')
    meas_interp = interpolate.interp1d(meas_wavelengths, meas_QE, kind='cubic')

    aligned_data = {
        'wavelengths': common_wavelengths,
        'sim_QE': sim_interp(common_wavelengths),
        'meas_QE': meas_interp(common_wavelengths)
    }

    # 표준편차도 보간
    if meas_std is not None:
        std_interp = interpolate.interp1d(meas_wavelengths, meas_std, kind='linear')
        aligned_data['meas_std'] = std_interp(common_wavelengths)

    return aligned_data

# 사용 예
aligned_R = align_wavelengths(
    sim_wavelengths, simulated_QE['R'],
    measured_data['wavelengths'], measured_data['R'],
    measured_data['R_std']
)
```

#### 2.3 정량적 비교 메트릭

```python
def calculate_comparison_metrics(sim_QE, meas_QE, meas_std=None):
    """
    시뮬레이션과 실측 간 비교 메트릭 계산

    Returns:
    --------
    metrics : dict
        다양한 통계적 메트릭
    """
    sim = np.array(sim_QE)
    meas = np.array(meas_QE)
    residuals = sim - meas

    metrics = {}

    # 1. Root Mean Square Error (RMSE)
    metrics['RMSE'] = np.sqrt(np.mean(residuals**2))

    # 2. Normalized RMSE (%)
    metrics['NRMSE'] = (metrics['RMSE'] / np.mean(meas)) * 100

    # 3. Mean Absolute Error (MAE)
    metrics['MAE'] = np.mean(np.abs(residuals))

    # 4. Mean Absolute Percentage Error (MAPE)
    metrics['MAPE'] = np.mean(np.abs(residuals / meas)) * 100

    # 5. R-squared (결정계수)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((meas - np.mean(meas))**2)
    metrics['R_squared'] = 1 - (ss_res / ss_tot)

    # 6. Pearson 상관계수
    metrics['Pearson_r'] = np.corrcoef(sim, meas)[0, 1]

    # 7. Maximum absolute error
    metrics['Max_error'] = np.max(np.abs(residuals))

    # 8. Chi-squared (표준편차가 있는 경우)
    if meas_std is not None:
        chi_squared = np.sum((residuals / meas_std)**2)
        metrics['Chi_squared'] = chi_squared
        metrics['Reduced_chi_squared'] = chi_squared / (len(sim) - 1)

    # 9. Bias (평균 편차)
    metrics['Bias'] = np.mean(residuals)

    return metrics

# 모든 픽셀에 대해 메트릭 계산
all_metrics = {}
for color in ['R', 'G', 'B']:
    aligned = align_wavelengths(
        sim_wavelengths, simulated_QE[color],
        measured_data['wavelengths'], measured_data[color],
        measured_data[f'{color}_std']
    )

    all_metrics[color] = calculate_comparison_metrics(
        aligned['sim_QE'],
        aligned['meas_QE'],
        aligned.get('meas_std')
    )

# 메트릭 출력
print("\n" + "="*60)
print("Comparison Metrics Summary")
print("="*60)
for color in ['R', 'G', 'B']:
    print(f"\n{color} Pixel:")
    metrics = all_metrics[color]
    print(f"  RMSE:          {metrics['RMSE']:.4f}")
    print(f"  NRMSE:         {metrics['NRMSE']:.2f}%")
    print(f"  MAE:           {metrics['MAE']:.4f}")
    print(f"  MAPE:          {metrics['MAPE']:.2f}%")
    print(f"  R²:            {metrics['R_squared']:.4f}")
    print(f"  Pearson r:     {metrics['Pearson_r']:.4f}")
    print(f"  Max error:     {metrics['Max_error']:.4f}")
    print(f"  Bias:          {metrics['Bias']:.4f}")
    if 'Reduced_chi_squared' in metrics:
        print(f"  χ²_reduced:    {metrics['Reduced_chi_squared']:.2f}")
```

#### 2.4 상세 시각화

```python
def plot_detailed_comparison(aligned_data, color_name, metrics, save_path=None):
    """
    실측-시뮬레이션 상세 비교 플롯

    4-패널 플롯:
    1. QE 스펙트럼 비교
    2. 절대 오차
    3. 상대 오차 (%)
    4. Residual 분포
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    wavelengths = aligned_data['wavelengths']
    sim_QE = aligned_data['sim_QE']
    meas_QE = aligned_data['meas_QE']
    meas_std = aligned_data.get('meas_std')

    residuals = sim_QE - meas_QE

    # 색상 설정
    colors_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
    pcolor = colors_map.get(color_name, 'black')

    # 1. QE 스펙트럼 비교
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(wavelengths*1000, sim_QE, '-', color=pcolor,
             linewidth=2.5, label='Simulation', alpha=0.8)
    ax1.plot(wavelengths*1000, meas_QE, 'o', color=pcolor,
             markersize=8, label='Measurement', markeredgecolor='black',
             markeredgewidth=1.5)

    if meas_std is not None:
        ax1.fill_between(wavelengths*1000,
                        meas_QE - meas_std,
                        meas_QE + meas_std,
                        alpha=0.2, color=pcolor, label='Measurement ±σ')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantum Efficiency', fontsize=12, fontweight='bold')
    ax1.set_title(f'{color_name} Pixel: QE Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(np.max(sim_QE), np.max(meas_QE)) * 1.1])

    # 메트릭 텍스트 박스
    textstr = '\n'.join([
        f"RMSE = {metrics['RMSE']:.4f}",
        f"NRMSE = {metrics['NRMSE']:.2f}%",
        f"R² = {metrics['R_squared']:.4f}",
        f"MAPE = {metrics['MAPE']:.2f}%"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', bbox=props)

    # 2. 절대 오차
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(wavelengths*1000, residuals, 'o-', color=pcolor,
            linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.fill_between(wavelengths*1000, 0, residuals, alpha=0.3, color=pcolor)
    ax2.set_xlabel('Wavelength (nm)', fontsize=11)
    ax2.set_ylabel('Absolute Error\n(Sim - Meas)', fontsize=11)
    ax2.set_title('Absolute Residuals', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # MAE 표시
    mae_line = np.full_like(wavelengths, metrics['MAE'])
    ax2.plot(wavelengths*1000, mae_line, 'r--', linewidth=1.5,
            label=f"MAE = {metrics['MAE']:.4f}")
    ax2.plot(wavelengths*1000, -mae_line, 'r--', linewidth=1.5)
    ax2.legend(fontsize=9)

    # 3. 상대 오차 (%)
    ax3 = fig.add_subplot(gs[1, 1])
    relative_error = (residuals / meas_QE) * 100
    ax3.plot(wavelengths*1000, relative_error, 's-', color=pcolor,
            linewidth=2, markersize=6)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.fill_between(wavelengths*1000, 0, relative_error, alpha=0.3, color=pcolor)
    ax3.set_xlabel('Wavelength (nm)', fontsize=11)
    ax3.set_ylabel('Relative Error (%)', fontsize=11)
    ax3.set_title('Relative Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # ±10% 범위 표시
    ax3.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.text(wavelengths[0]*1000, 10, '  +10%', fontsize=9, color='orange')

    # 4. Residual 히스토그램 및 정규분포 검정
    ax4 = fig.add_subplot(gs[2, 0])
    n, bins, patches = ax4.hist(residuals, bins=15, density=True,
                                alpha=0.7, color=pcolor, edgecolor='black')

    # 정규분포 피팅
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2),
            'k-', linewidth=2, label=f'Normal\nμ={mu:.4f}\nσ={sigma:.4f}')

    ax4.set_xlabel('Residual Value', fontsize=11)
    ax4.set_ylabel('Probability Density', fontsize=11)
    ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 5. Scatter plot (실측 vs 시뮬레이션)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(meas_QE, sim_QE, s=80, alpha=0.7, color=pcolor,
               edgecolors='black', linewidth=1.5)

    # 1:1 라인
    min_val = min(np.min(meas_QE), np.min(sim_QE))
    max_val = max(np.max(meas_QE), np.max(sim_QE))
    ax5.plot([min_val, max_val], [min_val, max_val],
            'k--', linewidth=2, label='1:1 Line')

    # 선형 회귀
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(meas_QE, sim_QE)
    line = slope * meas_QE + intercept
    ax5.plot(meas_QE, line, 'r-', linewidth=2,
            label=f'Fit: y={slope:.3f}x+{intercept:.3f}')

    ax5.set_xlabel('Measured QE', fontsize=11)
    ax5.set_ylabel('Simulated QE', fontsize=11)
    ax5.set_title('Correlation Plot', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_aspect('equal')

    # R² 표시
    ax5.text(0.05, 0.95, f"R² = {metrics['R_squared']:.4f}",
            transform=ax5.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved detailed comparison plot to {save_path}")

    return fig

# 모든 픽셀에 대해 상세 플롯 생성
for color in ['R', 'G', 'B']:
    aligned = align_wavelengths(
        sim_wavelengths, simulated_QE[color],
        measured_data['wavelengths'], measured_data[color],
        measured_data.get(f'{color}_std')
    )

    fig = plot_detailed_comparison(
        aligned, color,
        all_metrics[color],
        save_path=f'detailed_comparison_{color}_pixel.png'
    )
    plt.show()
```

#### 2.5 통계적 유의성 검정

```python
from scipy import stats

def statistical_significance_test(sim_QE, meas_QE, meas_std=None):
    """
    시뮬레이션과 실측 간 통계적 유의성 검정

    Returns:
    --------
    results : dict
        통계적 검정 결과
    """
    results = {}

    # 1. Paired t-test
    # H0: 시뮬레이션과 실측의 평균이 같다
    t_stat, p_value = stats.ttest_rel(sim_QE, meas_QE)
    results['t_test'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    # 2. Wilcoxon signed-rank test (비모수 검정)
    w_stat, p_value = stats.wilcoxon(sim_QE, meas_QE)
    results['wilcoxon_test'] = {
        'w_statistic': w_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    # 3. Kolmogorov-Smirnov test (분포 비교)
    ks_stat, p_value = stats.ks_2samp(sim_QE, meas_QE)
    results['ks_test'] = {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    # 4. Residual 정규성 검정 (Shapiro-Wilk)
    residuals = sim_QE - meas_QE
    sw_stat, p_value = stats.shapiro(residuals)
    results['shapiro_test'] = {
        'statistic': sw_stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05  # p > 0.05이면 정규분포
    }

    return results

# 통계 검정 수행
print("\n" + "="*60)
print("Statistical Significance Tests")
print("="*60)

for color in ['R', 'G', 'B']:
    aligned = align_wavelengths(
        sim_wavelengths, simulated_QE[color],
        measured_data['wavelengths'], measured_data[color]
    )

    test_results = statistical_significance_test(
        aligned['sim_QE'],
        aligned['meas_QE']
    )

    print(f"\n{color} Pixel:")
    print(f"  t-test p-value:     {test_results['t_test']['p_value']:.4f} " +
          f"{'(significant)' if test_results['t_test']['significant'] else ''}")
    print(f"  Wilcoxon p-value:   {test_results['wilcoxon_test']['p_value']:.4f}")
    print(f"  KS test p-value:    {test_results['ks_test']['p_value']:.4f}")
    print(f"  Residual normality: {test_results['shapiro_test']['p_value']:.4f} " +
          f"{'(normal)' if test_results['shapiro_test']['is_normal'] else '(non-normal)'}")
```

### 3. 민감도 분석

```python
def sensitivity_analysis(base_params, param_variations):
    """
    파라미터 변화에 대한 민감도 분석

    Parameters:
    -----------
    base_params : dict
        기준 파라미터
    param_variations : dict
        {'param_name': [values]}

    Returns:
    --------
    sensitivity_results : dict
    """
    results = {}

    for param_name, values in param_variations.items():
        QE_variations = []

        for value in values:
            # 파라미터 업데이트
            params = base_params.copy()
            params[param_name] = value

            # 시뮬레이션 실행
            obj = setup_sensor_simulation_with_params(params)
            QE = calculate_quantum_efficiency(obj, layer_PD, wavelength=0.55)
            QE_variations.append(QE)

        results[param_name] = (values, QE_variations)

    return results

# 민감도 분석 실행
param_variations = {
    'lens_sag': np.linspace(0.2, 0.5, 10),
    'filter_thickness': np.linspace(0.4, 0.8, 10),
    'metal_aperture': np.linspace(0.5, 0.9, 10),
    'PD_thickness': np.linspace(1.5, 3.0, 10)
}

sensitivity_results = sensitivity_analysis(base_params, param_variations)

# 플롯
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (param_name, (values, QEs)) in enumerate(sensitivity_results.items()):
    axes[i].plot(values, QEs, 'o-', linewidth=2)
    axes[i].set_xlabel(param_name)
    axes[i].set_ylabel('QE @ 550nm')
    axes[i].set_title(f'Sensitivity to {param_name}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=150)
```

## 고급 시나리오

### 1. 경사 입사 분석

```python
def analyze_angle_dependence(angles_deg):
    """입사각 의존성 분석"""
    angles_rad = angles_deg * np.pi / 180
    QE_vs_angle = {'R': [], 'G': [], 'B': []}

    for theta in angles_rad:
        obj = grcwa.obj(nG=201, L1=L1, L2=L2, freq,
                       theta=theta, phi=0, verbose=0)

        # 레이어 설정...

        # QE 계산
        QE = analyze_pixel_QE(obj, [0.55])  # 녹색광만

        for color in ['R', 'G', 'B']:
            QE_vs_angle[color].append(QE[color][0])

    # Chief Ray Angle (CRA) 분석
    plt.figure(figsize=(8, 6))
    for color in ['R', 'G', 'B']:
        plt.plot(angles_deg, QE_vs_angle[color],
                'o-', label=f'{color} pixel', linewidth=2)

    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('QE @ 550nm')
    plt.title('Angular Response (CRA Performance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('CRA_analysis.png', dpi=150)

    return QE_vs_angle

# CRA 분석 (일반적으로 0-30도)
angles = np.linspace(0, 30, 15)
CRA_results = analyze_angle_dependence(angles)
```

### 2. 크로스토크 분석

```python
def analyze_crosstalk(obj):
    """
    인접 픽셀 간 광학 크로스토크 분석

    Returns:
    --------
    crosstalk_matrix : ndarray
        크로스토크 행렬 [target_pixel, source_pixel]
    """
    # 포토다이오드 층에서 장 분포
    [Ex, Ey, Ez], _ = obj.Solve_FieldOnGrid(4, 1.0, [400, 400])
    intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

    # 4개 픽셀 영역 정의
    Nx, Ny = intensity.shape
    pixels = {
        'G1': intensity[:Nx//2, :Ny//2],
        'R':  intensity[Nx//2:, :Ny//2],
        'B':  intensity[:Nx//2, Ny//2:],
        'G2': intensity[Nx//2:, Ny//2:]
    }

    # 각 픽셀의 총 광량
    total_power = {name: np.sum(region) for name, region in pixels.items()}

    # 크로스토크 계산
    # 예: G1 픽셀에서 시작한 빛이 다른 픽셀에 도달한 비율
    crosstalk = {}
    for target in pixels.keys():
        crosstalk[target] = {}
        for source in pixels.keys():
            if target == source:
                crosstalk[target][source] = 1.0  # 자기 자신
            else:
                # 실제로는 여기 위치를 변경하여 계산 필요
                # 여기서는 근사적으로 경계 영역의 누설 계산
                crosstalk[target][source] = calculate_leakage(
                    pixels[source], pixels[target]
                )

    print("Crosstalk Matrix:")
    print("(From source pixel to target pixel)")
    for target in ['G1', 'R', 'B', 'G2']:
        print(f"{target}: ", end='')
        for source in ['G1', 'R', 'B', 'G2']:
            print(f"{crosstalk[target][source]:.3f} ", end='')
        print()

    return crosstalk

def calculate_leakage(source_region, target_region):
    """
    소스 픽셀에서 타겟 픽셀로의 광 누설 계산
    (간단한 근사 - 실제로는 더 정교한 분석 필요)
    """
    # 경계 영역의 강도 비교
    # 실제 구현은 시뮬레이션 설정에 따라 다름
    leakage = 0.01  # 예시: 1% 크로스토크
    return leakage

crosstalk_matrix = analyze_crosstalk(obj)
```

## 최적화 예제

### 마이크로렌즈 형상 최적화

```python
import grcwa
grcwa.set_backend('autograd')  # 경사도 계산 활성화
import autograd.numpy as np
from autograd import grad

def objective_function(lens_params_array):
    """
    최적화 목적 함수: QE 최대화

    Parameters:
    -----------
    lens_params_array : array
        [lens_radius, lens_sag, ...]

    Returns:
    --------
    -QE : float (음수 - 최소화 문제로 변환)
    """
    radius, sag = lens_params_array

    # 제약 조건
    if radius < 0.5 or radius > 2.0:
        return 1e6
    if sag < 0.1 or sag > 0.8:
        return 1e6

    # 시뮬레이션 설정
    lens_params = {'radius': radius, 'sag': sag,
                  'pitch': 2.24, 'n_lens': 1.55}

    obj = setup_sensor_with_lens(lens_params)
    QE = calculate_quantum_efficiency(obj, layer_PD=4, wavelength=0.55)

    return -QE  # 최대화 -> 최소화

# 경사도 계산
grad_objective = grad(objective_function)

# 최적화 (scipy 또는 nlopt 사용)
from scipy.optimize import minimize

initial_params = np.array([1.5, 0.3])  # [radius, sag]

result = minimize(
    objective_function,
    initial_params,
    method='L-BFGS-B',
    jac=grad_objective,
    bounds=[(0.5, 2.0), (0.1, 0.8)]
)

print("Optimized parameters:")
print(f"  Lens radius: {result.x[0]:.3f} μm")
print(f"  Lens sag:    {result.x[1]:.3f} μm")
print(f"  Maximum QE:  {-result.fun:.4f}")
```

## 체크리스트

시뮬레이션 정합성을 보장하기 위한 체크리스트:

- [ ] **단위 셀 선택**: 최소 반복 단위 확인
- [ ] **레이어 순서**: 실제 스택과 일치
- [ ] **재질 특성**: 파장 의존 굴절률/흡수 반영
- [ ] **그리드 해상도**: 수렴 테스트 완료 (Nx ≥ 200)
- [ ] **절단 차수**: nG 수렴 확인 (보통 201-301)
- [ ] **경계 조건**: 주기 경계가 타당한지 확인
- [ ] **에너지 보존**: R + T + A ≈ 1 (오차 < 1%)
- [ ] **실측 비교**: 가능하면 실험 데이터와 검증
- [ ] **민감도 분석**: 주요 파라미터 영향 파악
- [ ] **문서화**: 모든 가정과 근사 기록

## 참고 자료

### 이미지 센서 광학 설계

1. **R. Fontaine**, "The State-of-the-Art of Mainstream CMOS Image Sensors," *Proc. Int. Image Sensor Workshop* (2015)

2. **J. Nakamura**, "Image Sensors and Signal Processing for Digital Still Cameras," *CRC Press* (2005)

3. **Sony IMX Sensor Series Datasheets** - 실제 센서 스펙

### RCWA 응용

4. **Y. Kanamori et al.**, "Design of microlens for CMOS image sensor using RCWA," *Opt. Express* (2012)

5. **M. G. Moharam et al.**, "Rigorous coupled-wave analysis of metallic surface-relief gratings," *JOSA A* (1986)

## 다음 단계

- **[기본 개념](concepts.md)**: RCWA 기초
- **[API 레퍼런스](../api/core.md)**: 함수 상세 설명
- **[예제 갤러리](../examples/gallery.md)**: 다른 응용 예제
