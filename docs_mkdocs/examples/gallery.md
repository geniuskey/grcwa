# 예제 갤러리

GRCWA를 사용한 RCWA 시뮬레이션 예제를 둘러보세요.

## 기본 예제

### 예제 1: 원형 홀이 있는 정사각 격자

정사각 격자의 원형 공기 홀이 있는 광결정 슬랩을 통한 투과 및 반사 계산.

**구조:**

- 정사각 격자: 주기 = 1.5 μm
- 원형 공기 홀이 있는 실리콘 슬랩 (ε=4)
- 홀 반경: 0.3 × 주기
- 슬랩 두께: 0.2 μm

**결과:**

- 총 및 차수별 반사/투과
- 다중 차수로의 회절 시연

---

### 예제 2: 두 개의 패턴 레이어

두 개의 서로 다른 패턴 레이어가 있는 다층 구조.

**구조:**

- 레이어 1: 원형 홀 (ε=4)
- 레이어 2: 정사각 홀 (ε=6)
- 각 레이어의 다른 패턴

**결과:**

- 다중 패턴 레이어 처리 방법 표시
- 경사 입사 (θ = π/10)

---

### 예제 3: 위상 최적화

자동 미분을 사용한 역설계.

**목표:** 단일 패턴 레이어로부터 반사 최대화

**방법:**

- 경사도를 위한 Autograd
- 최적화를 위한 NLOPT
- 경사도 기반 위상 최적화

**결과:**

- 최적화된 유전 패턴
- 반복에 따른 반사의 수렴

---

### 예제 4: 육각 격자

원형 홀의 육각 격자.

**구조:**

- 육각 격자 (60° 각도)
- 원형 공기 홀
- 높은 그리드 해상도 (1000×1000)

**결과:**

- 비직교 격자 시연
- 육각 대칭을 위한 적절한 좌표 변환

---

## 고급 예제

### Bragg 거울

다층 Bragg 반사체 (1D 광결정).

```python
import grcwa
import numpy as np

# 매개변수
wavelength = 1.0
freq = 1.0 / wavelength
n1, n2 = 2.0, 1.5  # 고/저 굴절률
d1 = wavelength / (4*n1)  # 사분파장 두께
d2 = wavelength / (4*n2)
N_pairs = 10  # 레이어 쌍 수

# 설정
obj = grcwa.obj(51, [1,0], [0,1], freq, 0, 0, verbose=0)

# 레이어 추가
for i in range(N_pairs):
    obj.Add_LayerUniform(d1, n1**2)
    obj.Add_LayerUniform(d2, n2**2)

obj.Init_Setup()
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
R, T = obj.RT_Solve(normalize=1)

print(f"Bragg 거울: R={R:.4f}, T={T:.4f}")
```

**결과:** 설계 파장에서 높은 반사 (~99%).

---

### 반사 방지 코팅

사분파장 반사 방지 코팅.

```python
# 기판: n=3.5 (예: GaAs)
# 코팅: n=sqrt(3.5) ≈ 1.87 (최적)
# 두께: λ/4n

wavelength = 1.0
n_substrate = 3.5
n_coating = np.sqrt(n_substrate)
thickness = wavelength / (4*n_coating)

obj = grcwa.obj(51, [1,0], [0,1], 1/wavelength, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)  # 공기
obj.Add_LayerUniform(thickness, n_coating**2)  # AR 코팅
obj.Add_LayerUniform(10.0, n_substrate**2)  # 기판 (두꺼움)

obj.Init_Setup()
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
R, T = obj.RT_Solve(normalize=1)

print(f"AR 코팅 없이, R은: {((1-n_substrate)/(1+n_substrate))**2:.4f}")
print(f"AR 코팅 사용: R={R:.6f}")
```

**결과:** 설계 파장에서 거의 0에 가까운 반사.

---

### 메타표면 렌즈

빔 조향을 위한 위상 경사 메타표면.

```python
# 단위 셀에 위상 경사 생성
def phase_gradient_pattern(Nx, Ny, gradient_angle):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 위상 = k*x*sin(theta), 유전체 변화로 근사
    # 단순화된 모델: 크기가 변하는 기둥 사용
    phase = 2*np.pi*X*np.tan(gradient_angle)
    pillar_radius = 0.1 + 0.2 * (phase / (2*np.pi)) % 1

    eps = np.ones((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if (X[i,j]-0.5)**2 + (Y[i,j]-0.5)**2 < pillar_radius**2:
                eps[i,j] = 12.0  # 실리콘

    return eps

# 시뮬레이션
gradient_angle = 10 * np.pi/180
eps_meta = phase_gradient_pattern(300, 300, gradient_angle)

obj = grcwa.obj(201, [0.6,0], [0,0.6], 1.0, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(0.5, 300, 300)
obj.Add_LayerUniform(1.0, 1.0)
obj.Init_Setup()
obj.GridLayer_geteps(eps_meta.flatten())
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

# 주요 회절 차수 찾기
max_order = np.argmax(Ti)
print(f"차수 {obj.G[max_order]}에서 최대 파워: T={Ti[max_order]:.4f}")
```

---

### 광결정 도파관

광결정의 선 결함.

```python
# 선 결함이 있는 광결정 생성
def pc_waveguide(Nx, Ny, lattice_const=0.4, hole_radius=0.3):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    eps = np.ones((Nx, Ny)) * 12.0  # 실리콘 배경

    # 규칙적인 홀 배열
    for i in range(-2, 3):
        for j in range(-2, 3):
            if j == 0:
                continue  # 중간 행 건너뛰기 (도파관)
            cx = 0.5 + i*lattice_const
            cy = 0.5 + j*lattice_const
            hole = (X-cx)**2 + (Y-cy)**2 < hole_radius**2
            eps[hole] = 1.0

    return eps

eps_wg = pc_waveguide(400, 400)

# 밴드갭 주파수에서 가이드 모드 시뮬레이션
obj = grcwa.obj(201, [2.0,0], [0,2.0], 0.3, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(1.0, 400, 400)
obj.Add_LayerUniform(1.0, 1.0)
obj.Init_Setup()
obj.GridLayer_geteps(eps_wg.flatten())
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

R, T = obj.RT_Solve(normalize=1)
print(f"가이드 모드: R={R:.4f}, T={T:.4f}")
```

---

### 격자 결합기

수직 입사광을 면내 도파관 모드로 결합.

```python
# 1D 격자
def grating(Nx, Ny, period_frac=0.5, depth_frac=0.3):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 배경: 실리콘
    eps = np.ones((Nx, Ny)) * 12.0

    # 격자: 교대로 실리콘 제거
    grating_lines = (X % (1/5)) < (period_frac / 5)
    eps[grating_lines] = 1.0

    return eps

eps_grating = grating(400, 200)

obj = grcwa.obj(201, [0.6,0], [0,0.3], 1.0, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(0.2, 400, 200)
obj.Add_LayerUniform(1.0, 12.0)  # 실리콘 기판
obj.Init_Setup()
obj.GridLayer_geteps(eps_grating.flatten())
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

# 다른 차수로 결합된 파워
for i in range(min(10, obj.nG)):
    if Ti[i] > 0.01:
        print(f"차수 {obj.G[i]}: T={Ti[i]:.4f}")
```

---

## 스펙트럼 계산

### 광결정 밴드 다이어그램

주파수 및 입사각에 따른 투과 계산.

```python
frequencies = np.linspace(0.3, 0.7, 40)
angles = np.linspace(0, 45, 30) * np.pi/180

T_map = np.zeros((len(frequencies), len(angles)))

for i, freq in enumerate(frequencies):
    for j, theta in enumerate(angles):
        obj = grcwa.obj(101, [1,0], [0,1], freq, theta, 0, verbose=0)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(0.5, 200, 200)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Init_Setup()
        obj.GridLayer_geteps(eps_pattern.flatten())
        obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
        R, T = obj.RT_Solve(normalize=1)
        T_map[i,j] = T

# 밴드 다이어그램 플롯
plt.figure(figsize=(8,6))
plt.pcolormesh(angles*180/np.pi, frequencies, T_map, cmap='hot', shading='auto')
plt.colorbar(label='투과')
plt.xlabel('입사각 (도)')
plt.ylabel('주파수 (c/λ)')
plt.title('광결정 밴드 다이어그램')
plt.show()
```

---

## 시각화 예제

### 장 강도 프로파일

```python
# 풀이 후
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(1, 0.15, [200, 200])

# 강도
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

# 플롯
plt.figure(figsize=(8,8))
plt.imshow(I.T, origin='lower', cmap='hot', extent=[0, L1[0], 0, L2[1]])
plt.colorbar(label='|E|²')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.title('전기장 강도')
plt.tight_layout()
plt.show()
```

### Poynting 벡터 장

```python
# Poynting 벡터 계산
Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))

# 플롯
plt.figure(figsize=(8,8))
plt.quiver(Sx[::10,::10], Sy[::10,::10])
plt.imshow(np.abs(Sz.T), origin='lower', cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Sz')
plt.title('Poynting 벡터 (에너지 흐름)')
plt.show()
```

---

## 나만의 예제 만들기 팁

1. **간단하게 시작**: 설정을 확인하기 위해 균일한 레이어로 시작
2. **수렴 테스트**: 결과가 안정될 때까지 $N_G$ 증가
3. **에너지 보존 확인**: 무손실의 경우 $R + T = 1$
4. **패턴 시각화**: `Return_eps()`를 사용하여 패턴 확인
5. **장 분석**: `Solve_FieldOnGrid()`를 사용하여 물리학 이해
6. **매개변수 스윕**: 파장, 각도 또는 기하학 변경

## 참고 자료

- **[튜토리얼](../tutorials/tutorial1.md)**: 단계별 가이드 예제
- **[API 레퍼런스](../api/core.md)**: 완전한 함수 문서
- **[FAQ](../reference/faq.md)**: 자주 묻는 질문
