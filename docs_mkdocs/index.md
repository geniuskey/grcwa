# GRCWA 문서에 오신 것을 환영합니다

<div align="center" markdown>

![RCWA 구조](../imag/scheme.png){ width="600" loading=lazy }

**자동미분 지원 광결정 엄밀 결합파 해석**

[:octicons-mark-github-16: GitHub](https://github.com/weiliangjinca/grcwa){ .md-button .md-button--primary }
[:fontawesome-brands-python: PyPI](https://pypi.org/project/grcwa/){ .md-button }
[:octicons-book-16: Read the Docs](https://grcwa.readthedocs.io){ .md-button }

</div>

---

## GRCWA란 무엇인가요?

!!! abstract "개요"
    **GRCWA** (autoGradable Rigorous Coupled Wave Analysis)는 주기적 광결정 구조와 빛의 상호작용을 시뮬레이션하는 강력한 Python 라이브러리입니다. 엄밀 결합파 해석(RCWA) 방법을 구현하며, 자동 미분 기능을 완벽하게 지원하여 광학 소자의 역설계 및 최적화에 이상적입니다.

## :material-features: 주요 기능

### :microscope: 물리 기반 시뮬레이션

<div class="grid cards" markdown>

-   :fontawesome-solid-wave-square: __엄밀한 시뮬레이션__

    ---

    Fourier 모드 방법을 사용하여 Maxwell 방정식을 정확하게 풉니다

    - 완전한 벡터장 계산
    - 임의의 2D 주기 구조
    - 다층 구조 적층

-   :material-layers-triple: __다층 구조__

    ---

    복잡한 다층 구조를 지원합니다

    - 독립적인 유전 프로파일
    - 균일 및 패턴 레이어
    - 무제한 레이어 개수

</div>

### :dart: 임의의 형상

=== "균일 레이어"

    일정한 유전율을 가진 단순한 유전체 슬랩

    ```python
    obj.Add_LayerUniform(thickness=0.5, epsilon=4.0)
    ```

=== "그리드 기반 패턴"

    직교 좌표계 그리드로 정의되는 임의의 2D 패턴

    ```python
    obj.Add_LayerGrid(thickness=0.3, Nx=400, Ny=400)
    obj.GridLayer_geteps(epsilon_grid.flatten())
    ```

=== "해석적 Fourier"

    알려진 Fourier 급수를 가진 형상에 효율적

    ```python
    obj.Add_LayerFourier(thickness=0.2, params)
    ```

### :rocket: 자동 미분

!!! tip "경사도 기반 최적화"
    자동 경사도 계산을 위해 [Autograd](https://github.com/HIPS/autograd)와 통합되었습니다

**자동미분 가능한 매개변수:**

- [x] 모든 그리드 점의 유전상수
- [x] 레이어 두께
- [x] 작동 주파수
- [x] 입사각
- [x] 격자 주기

### :triangular_ruler: 유연한 격자

<div class="grid cards" markdown>

-   __정사각__

    ```python
    L1 = [a, 0]
    L2 = [0, a]
    ```

-   __육각__

    ```python
    L1 = [a, 0]
    L2 = [a/2, a*√3/2]
    ```

-   __임의__

    ```python
    L1 = [Lx1, Ly1]
    L2 = [Lx2, Ly2]
    ```

</div>

## GRCWA로 무엇을 할 수 있나요?

### 해석 작업
- 반사 및 투과 스펙트럼 계산
- 회절 차수 분석
- 실공간 및 Fourier 공간에서 전자기장 계산
- Poynting 플럭스 및 에너지 흐름 계산
- Maxwell 응력 텐서 평가

### 설계 및 최적화
- 광결정 구조의 위상 최적화
- 메타표면의 역설계
- 경사도 기반 최적화:
    - 광학 필터
    - 반사방지 코팅
    - 광결정 밴드갭 구조
    - 광대역 반사기
    - 고효율 광흡수체

### 연구 응용
- 광결정 설계
- 메타물질 공학
- 회절 격자 설계
- 회절 광학
- 태양전지 최적화
- 라이다(LIDAR) 부품 설계

## 빠른 예제

시작하기 위한 간단한 예제입니다:

```python
import grcwa
import numpy as np

# 격자 및 주파수 정의
L1 = [1.5, 0]  # 격자 벡터 1
L2 = [0, 1.5]  # 격자 벡터 2
freq = 1.0     # 주파수 (c=1)
theta = 0.0    # 입사각
phi = 0.0      # 방위각
nG = 101       # 절단 차수

# RCWA 객체 생성
obj = grcwa.obj(nG, L1, L2, freq, theta, phi)

# 레이어 추가: 진공 + 패턴 + 진공
obj.Add_LayerUniform(1.0, 1.0)        # 진공 레이어
obj.Add_LayerGrid(0.2, 400, 400)       # 패턴 레이어
obj.Add_LayerUniform(1.0, 1.0)        # 진공 레이어

# 역격자 설정
obj.Init_Setup()

# 패턴 정의 (원형 홀)
Nx, Ny = 400, 400
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
pattern = np.ones((Nx, Ny)) * 4.0  # 실리콘 (ε=4)
hole = (X-0.5)**2 + (Y-0.5)**2 < 0.3**2
pattern[hole] = 1.0  # 공기 홀

# 패턴 입력
obj.GridLayer_geteps(pattern.flatten())

# 여기 설정 (p-편광 평면파)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0, order=0)

# 반사 및 투과 계산
R, T = obj.RT_Solve(normalize=1)
print(f'R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}')
```

## 왜 GRCWA를 선택해야 하나요?

| 기능 | GRCWA | 전통적 RCWA |
|---------|-------|------------------|
| 자동 미분 | ✅ 내장 | ❌ 수동 유도 |
| 최적화 준비 | ✅ 직접 통합 | ❌ 외부 도구 필요 |
| Python 네이티브 | ✅ 사용하기 쉬움 | ⚠️ 주로 C/Fortran |
| 임의 패턴 | ✅ 그리드 기반 | ⚠️ 제한된 형상 |
| 활발한 개발 | ✅ 오픈소스 | ⚠️ 다양함 |

## 시작하기

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __빠른 시작__

    ---

    빠른 시작 가이드로 몇 분 안에 시작하세요

    [:octicons-arrow-right-24: 빠른 시작](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } __이론 학습__

    ---

    RCWA의 물리학과 수학 이해하기

    [:octicons-arrow-right-24: 이론](theory/principles.md)

-   :material-code-braces:{ .lg .middle } __API 레퍼런스__

    ---

    모든 클래스와 함수의 상세 문서

    [:octicons-arrow-right-24: API 문서](api/core.md)

-   :material-school:{ .lg .middle } __튜토리얼__

    ---

    일반적인 사용 사례를 위한 단계별 튜토리얼

    [:octicons-arrow-right-24: 튜토리얼](tutorials/tutorial1.md)

</div>

## 프로젝트 정보

- **저자**: Weiliang Jin (jwlaaa@gmail.com)
- **버전**: 0.1.2
- **라이선스**: GPL v3
- **Python**: ≥ 3.5
- **저장소**: [github.com/weiliangjinca/grcwa](https://github.com/weiliangjinca/grcwa)

## 인용

연구에서 GRCWA를 사용하는 경우 다음을 인용해 주세요:

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

## 도움이 필요하신가요?

- 📖 [상세 문서](introduction.md) 읽기
- 💡 [예제](examples/gallery.md) 확인하기
- 🐛 [GitHub](https://github.com/weiliangjinca/grcwa/issues)에 이슈 보고하기
- 📧 저자에게 이메일 보내기: jwlaaa@gmail.com
