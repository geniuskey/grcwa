# RCWA의 수학적 공식화

이 페이지는 RCWA의 수학적 프레임워크를 상세히 유도합니다.

## 장 전개

### Fourier 급수 전개

격자 벡터 $\mathbf{L}_1, \mathbf{L}_2$를 갖는 주기 구조의 경우, 모든 주기 함수는 다음과 같이 전개할 수 있습니다:

$$
f(\mathbf{r}_\parallel) = \sum_{m,n} f_{mn} \exp(i\mathbf{G}_{mn} \cdot \mathbf{r}_\parallel)
$$

여기서 $\mathbf{G}_{mn} = m\mathbf{K}_1 + n\mathbf{K}_2$는 역격자 벡터입니다.

### 장 성분

각 레이어에서 전기장과 자기장은:

$$
\mathbf{E}(\mathbf{r}) = \sum_{mn} \mathbf{E}_{mn}(z) \exp(i\mathbf{k}_{mn,\parallel} \cdot \mathbf{r}_\parallel)
$$

$$
\mathbf{H}(\mathbf{r}) = \sum_{mn} \mathbf{H}_{mn}(z) \exp(i\mathbf{k}_{mn,\parallel} \cdot \mathbf{r}_\parallel)
$$

여기서 $\mathbf{k}_{mn,\parallel} = \mathbf{k}_{\parallel,0} + \mathbf{G}_{mn}$입니다.

## 패턴 레이어의 고유값 문제

### Fourier 공간의 Maxwell 방정식

회전 방정식에 Fourier 변환 적용:

$$
\nabla \times \mathbf{E} = i\omega\mu\mathbf{H}
$$

$$
\nabla \times \mathbf{H} = -i\omega\varepsilon\mathbf{E}
$$

TM 모드(p-편광)의 경우 $\mathbf{H}$를 소거하면:

$$
\nabla \times (\varepsilon^{-1} \nabla \times \mathbf{H}) = \omega^2 \mu \mathbf{H}
$$

TE 모드(s-편광)의 경우:

$$
\nabla \times (\mu^{-1} \nabla \times \mathbf{E}) = \omega^2 \varepsilon \mathbf{E}
$$

### 접선 장 공식화

경계면을 가로질러 연속이므로 접선 장 성분 $(E_x, E_y, H_x, H_y)$로 작업합니다.

벡터 정의:

$$
\mathbf{E}_\parallel = \begin{pmatrix} E_x \\ E_y \end{pmatrix}, \quad
\mathbf{H}_\parallel = \begin{pmatrix} H_x \\ H_y \end{pmatrix}
$$

Maxwell 방정식으로부터:

$$
\frac{\partial}{\partial z} \mathbf{E}_\parallel = i\omega\mu \hat{z} \times \mathbf{H}_\parallel - i\mathbf{k}_\parallel E_z
$$

$$
\frac{\partial}{\partial z} \mathbf{H}_\parallel = -i\omega\varepsilon \hat{z} \times \mathbf{E}_\parallel - i\mathbf{k}_\parallel H_z
$$

### 법선 성분 소거

$\nabla \cdot \mathbf{D} = 0$으로부터:

$$
i k_x E_x + i k_y E_y + \frac{\partial (\varepsilon E_z)}{\partial z} = 0
$$

각 레이어에서 z-불변 $\varepsilon$의 경우:

$$
E_z = -\frac{1}{\varepsilon}(k_x E_x + k_y E_y)
$$

마찬가지로 $\nabla \cdot \mathbf{B} = 0$으로부터:

$$
H_z = -\frac{1}{\mu}(k_x H_x + k_y H_y)
$$

### 행렬 공식화

$K_\perp$ 연산자 정의:

$$
K_\perp = \begin{pmatrix} k_y^2 & -k_x k_y \\ -k_x k_y & k_x^2 \end{pmatrix}
$$

이는 컨볼루션 행렬과 함께 Fourier 공간에 작용합니다.

결합파 방정식은:

$$
\frac{\partial}{\partial z} \begin{pmatrix} \mathbf{E}_\parallel \\ \mathbf{H}_\parallel \end{pmatrix}
= i \begin{pmatrix} 0 & A \\ B & 0 \end{pmatrix}
\begin{pmatrix} \mathbf{E}_\parallel \\ \mathbf{H}_\parallel \end{pmatrix}
$$

여기서 $A$와 $B$는 $\varepsilon$, $\mu$, $K_\perp$를 포함하는 행렬입니다.

### 고유값 문제

$\exp(iq z)$ 의존성을 대입:

$$
\begin{pmatrix} \mathbf{E}_\parallel \\ \mathbf{H}_\parallel \end{pmatrix}
= \begin{pmatrix} \boldsymbol{\phi}_E \\ \boldsymbol{\phi}_H \end{pmatrix} e^{iq z}
$$

이는 고유값 방정식을 제공합니다:

$$
\begin{pmatrix} 0 & A \\ B & 0 \end{pmatrix}
\begin{pmatrix} \boldsymbol{\phi}_E \\ \boldsymbol{\phi}_H \end{pmatrix}
= q \begin{pmatrix} \boldsymbol{\phi}_E \\ \boldsymbol{\phi}_H \end{pmatrix}
$$

또는 동등하게:

$$
AB \boldsymbol{\phi}_H = q^2 \boldsymbol{\phi}_H
$$

$$
BA \boldsymbol{\phi}_E = q^2 \boldsymbol{\phi}_E
$$

고유값 $q$는 고유모드의 파동 벡터 $z$-성분입니다.

## 컨볼루션 행렬

### 유전 Fourier 변환

유전 함수는 다음과 같이 전개됩니다:

$$
\varepsilon(x,y) = \sum_{mn} \varepsilon_{mn} \exp(i\mathbf{G}_{mn} \cdot \mathbf{r}_\parallel)
$$

Fourier 계수:

$$
\varepsilon_{mn} = \frac{1}{A_{\text{cell}}} \int_{\text{cell}} \varepsilon(x,y) \exp(-i\mathbf{G}_{mn} \cdot \mathbf{r}_\parallel) dx dy
$$

실제로는 2D FFT를 통해 계산:

$$
\varepsilon_{mn} = \text{FFT2D}[\varepsilon(x,y)]
$$

### Fourier 공간의 컨볼루션

실공간 곱셈 → Fourier 공간 컨볼루션:

$$
[\varepsilon(x,y) E_x(x,y,z)]_{mn} = \sum_{m'n'} \varepsilon_{m-m',n-n'} E_{x,m'n'}(z)
$$

이는 행렬 곱으로 표현됩니다:

$$
(\varepsilon \mathbf{E})_{mn} = \sum_{m'n'} [\mathcal{E}]_{mn,m'n'} E_{m'n'}
$$

여기서 $[\mathcal{E}]_{mn,m'n'} = \varepsilon_{m-m',n-n'}$는 **컨볼루션 행렬**입니다.

### $\varepsilon^{-1}$에 대한 Laurent 규칙

$\varepsilon^{-1}$을 포함하는 곱의 경우, $1/\varepsilon(x,y)$의 직접 Fourier 변환은:

$$
[\varepsilon^{-1}]_{mn} = \text{FFT2D}[1/\varepsilon(x,y)]
$$

이것이 **Laurent 규칙**이며 $\mathcal{E}$를 역행렬하는 것보다 $\varepsilon^{-1}$에 대해 더 나은 수렴을 제공합니다.

## S-행렬 알고리즘

### 레이어 전달 행렬

균일 또는 패턴 레이어 내에서 해는:

$$
\begin{pmatrix} \mathbf{E}_\parallel(z) \\ \mathbf{H}_\parallel(z) \end{pmatrix}
= \sum_j c_j^+ \begin{pmatrix} \boldsymbol{\phi}_{E,j} \\ \boldsymbol{\phi}_{H,j} \end{pmatrix} e^{iq_j z}
+ \sum_j c_j^- \begin{pmatrix} \boldsymbol{\phi}_{E,j} \\ -\boldsymbol{\phi}_{H,j} \end{pmatrix} e^{-iq_j z}
$$

여기서 $c_j^+$와 $c_j^-$는 전방 및 후방 모드 진폭입니다.

두 위치 $z_1$과 $z_2 = z_1 + d$에서:

$$
\begin{pmatrix} \mathbf{E}_\parallel(z_2) \\ \mathbf{H}_\parallel(z_2) \end{pmatrix}
= T \begin{pmatrix} \mathbf{E}_\parallel(z_1) \\ \mathbf{H}_\parallel(z_1) \end{pmatrix}
$$

여기서 $T$는 전달 행렬입니다.

### S-행렬 정의

**산란 행렬**(S-행렬)은 입사파와 출사파를 연결합니다:

$$
\begin{pmatrix} \mathbf{c}^+_{\text{out}} \\ \mathbf{c}^-_{\text{in}} \end{pmatrix}
= \begin{pmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix}
\begin{pmatrix} \mathbf{c}^+_{\text{in}} \\ \mathbf{c}^-_{\text{out}} \end{pmatrix}
$$

여기서:

- $S_{11}$: 위로부터의 반사
- $S_{22}$: 아래로부터의 반사
- $S_{12}$, $S_{21}$: 투과

### S-행렬 합성

S-행렬 $S^{(A)}$와 $S^{(B)}$를 갖는 두 레이어가 순차적으로 적층된 경우:

$$
S^{(AB)}_{11} = S^{(A)}_{11} + S^{(A)}_{12} (I - S^{(B)}_{11} S^{(A)}_{22})^{-1} S^{(B)}_{11} S^{(A)}_{21}
$$

$$
S^{(AB)}_{12} = S^{(A)}_{12} (I - S^{(B)}_{11} S^{(A)}_{22})^{-1} S^{(B)}_{12}
$$

$$
S^{(AB)}_{21} = S^{(B)}_{21} (I - S^{(A)}_{22} S^{(B)}_{11})^{-1} S^{(A)}_{21}
$$

$$
S^{(AB)}_{22} = S^{(B)}_{22} + S^{(B)}_{21} (I - S^{(A)}_{22} S^{(B)}_{11})^{-1} S^{(A)}_{22} S^{(B)}_{12}
$$

이를 통해 레이어별로 전체 S-행렬을 구축할 수 있습니다.

## 입력/출력에서의 경계 조건

### 입력 영역

$\varepsilon_{\text{in}}$을 갖는 반무한 균일 영역:

$$
\mathbf{E}^{\text{in}} = \mathbf{E}_{\text{inc}} e^{i\mathbf{k}_{\text{inc}} \cdot \mathbf{r}}
+ \sum_{mn} r_{mn} \mathbf{E}_{mn}^{\text{refl}} e^{i\mathbf{k}_{mn}^{\text{refl}} \cdot \mathbf{r}}
$$

여기서:

- $\mathbf{E}_{\text{inc}}$: 입사 장
- $r_{mn}$: 각 차수의 반사 계수

### 출력 영역

$\varepsilon_{\text{out}}$을 갖는 반무한 균일 영역:

$$
\mathbf{E}^{\text{out}} = \sum_{mn} t_{mn} \mathbf{E}_{mn}^{\text{trans}} e^{i\mathbf{k}_{mn}^{\text{trans}} \cdot \mathbf{r}}
$$

여기서 $t_{mn}$은 투과 계수입니다.

### 계수 풀이

상단 및 하단 경계면에 경계 조건 적용:

$$
\begin{pmatrix} \mathbf{r} \\ \mathbf{t} \end{pmatrix}
= S_{\text{total}} \begin{pmatrix} \mathbf{inc} \\ 0 \end{pmatrix}
$$

여기서 $S_{\text{total}}$은 전체 구조의 총 S-행렬입니다.

## 파워 계산

### 정규화 Poynting 플럭스

각 회절 차수 $(m,n)$에 대해 정규화된 파워는:

$$
P_{mn} = \frac{1}{2} \text{Re}\left( \frac{k_{z,mn}}{\omega\mu} \right) |A_{mn}|^2
$$

여기서 $A_{mn}$은 장 진폭입니다.

### 반사 및 투과

총 반사:

$$
R = \sum_{mn} \frac{k_{z,mn}^{\text{refl}} / \omega\mu_{\text{in}}}{k_{z,\text{inc}} / \omega\mu_{\text{in}}} |r_{mn}|^2
= \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{refl}})}{\text{Re}(k_{z,\text{inc}})} |r_{mn}|^2
$$

총 투과:

$$
T = \sum_{mn} \frac{k_{z,mn}^{\text{trans}} / \omega\mu_{\text{out}}}{k_{z,\text{inc}} / \omega\mu_{\text{in}}} |t_{mn}|^2
= \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{trans}}) \mu_{\text{in}}}{\text{Re}(k_{z,\text{inc}}) \mu_{\text{out}}} |t_{mn}|^2
$$

정규화는 손실 없는 구조에 대해 $R + T = 1$을 보장합니다.

## 장 재구성

### Fourier 공간 장

레이어 내 임의의 위치 $(x, y, z)$에서 장은:

$$
\mathbf{E}(x,y,z) = \sum_{mn} \mathbf{E}_{mn}(z) \exp(i k_{x,mn} x + i k_{y,mn} y)
$$

여기서 $\mathbf{E}_{mn}(z)$는 고유모드 전개로부터 계산됩니다.

### 실공간 장

실공간 그리드에서 장을 얻으려면 역 FFT 사용:

$$
\mathbf{E}(x_i, y_j, z) = \text{IFFT2D}[\mathbf{E}_{mn}(z)]
$$

### 부피 적분

부피 적분 계산(예: 에너지 밀도)의 경우:

$$
\int_V |\mathbf{E}|^2 dV = \sum_{mn} |\mathbf{E}_{mn}|^2 V_{\text{unit}}
$$

여기서 $V_{\text{unit}} = A_{\text{cell}} \cdot d$는 단위 셀 부피입니다.

## 수치 안정성

### 향상된 투과 행렬

두꺼운 레이어의 경우 직접 지수 $e^{iq_j d}$는 큰 $|q_j|$에 대해 오버플로할 수 있습니다.

**향상된 투과 행렬** 방법 사용:

- 가장 큰 $q_j$로 지수 스케일링
- 절대값이 아닌 비율 계산
- 오버플로/언더플로 방지

### 고유값 순서

고유값 $q_j$는 다음과 같이 순서를 정해야 합니다:

- 전방 전파: $\text{Re}(q) > 0$ 또는 $\text{Im}(q) < 0$
- 후방 전파: $\text{Re}(q) < 0$ 또는 $\text{Im}(q) > 0$

이는 S-행렬 알고리즘의 수치 안정성을 보장합니다.

### 행렬 역행렬

$\mathcal{E}$와 같은 행렬은 다음의 경우 ill-conditioned일 수 있습니다:

- 고대비 구조
- 큰 절단 차수
- 얇은 특징

직접 역행렬이 실패하면 정규화 또는 반복 솔버를 사용하세요.

## 수렴 가속

### 적응 좌표 변환

수직 측벽이 있는 격자의 경우 $u$-공간으로 매핑:

$$
u(x) = x + \sum_n a_n \sin(2\pi n x / \Lambda)
$$

이는 불연속성을 매끄럽게 하고 수렴을 개선합니다.

### 완전 정합 레이어 (PML)

흡수 경계의 경우(주기적 RCWA에서는 일반적으로 사용되지 않음):

$$
\varepsilon \to \varepsilon s, \quad \mu \to \mu s
$$

여기서 $s = 1 + i\sigma/\omega$는 PML 신축 매개변수입니다.

### 서브픽셀 스무딩

날카로운 특징의 경우 점 샘플링이 아닌 그리드 셀에 대해 $\varepsilon$를 평균:

$$
\varepsilon_{\text{cell}} = \frac{1}{A_{\text{cell}}} \int_{\text{cell}} \varepsilon(x,y) dx dy
$$

계단 근사에 대한 수렴을 개선합니다.

## 주요 방정식 요약

| 양 | 방정식 |
|----------|----------|
| **장 전개** | $\mathbf{E} = \sum_{mn} \mathbf{E}_{mn}(z) e^{i\mathbf{k}_{mn} \cdot \mathbf{r}_\parallel}$ |
| **파동 벡터** | $\mathbf{k}_{mn} = \mathbf{k}_{\parallel,0} + \mathbf{G}_{mn}$ |
| **고유값 문제** | $AB\boldsymbol{\phi} = q^2 \boldsymbol{\phi}$ |
| **S-행렬** | $\begin{pmatrix} \mathbf{out} \end{pmatrix} = S \begin{pmatrix} \mathbf{in} \end{pmatrix}$ |
| **반사** | $R = \sum_{mn} \frac{\text{Re}(k_{z,mn}^r)}{\text{Re}(k_{z,inc})} |r_{mn}|^2$ |
| **투과** | $T = \sum_{mn} \frac{\text{Re}(k_{z,mn}^t)}{\text{Re}(k_{z,inc})} |t_{mn}|^2$ |
| **에너지 보존** | $R + T = 1$ (무손실) |

## GRCWA에서의 구현

GRCWA는 다음에서 이러한 방정식을 구현합니다:

- **`rcwa.py`**: 주 솔버, S-행렬 알고리즘
- **`fft_funs.py`**: FFT를 통한 컨볼루션 행렬
- **`kbloch.py`**: 역격자 및 파동 벡터
- **`primitives.py`**: Autograd 호환 고유값 솔버

주요 최적화:

- 컨볼루션 행렬을 위한 빠른 FFT
- 안정성을 위한 향상된 투과 행렬
- 경사도 계산을 위한 Autograd 프리미티브

## 추가 자료

유도 및 증명에 대해서는 다음을 참조하세요:

- Moharam et al., "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings," JOSA A (1995)
- Liu and Fan, "Three-dimensional photonic crystals by total-internal reflection," Optics Letters (2005)
- Whittaker and Culshaw, "Scattering-matrix treatment of patterned multilayer photonic structures," Phys. Rev. B (1999)

## 다음 단계

- **[RCWA 알고리즘](algorithm.md)**: 단계별 계산 절차
- **[API 레퍼런스](../api/core.md)**: 방정식이 코드에 매핑되는 방법
- **[튜토리얼](../tutorials/tutorial1.md)**: 실제 이론 적용
