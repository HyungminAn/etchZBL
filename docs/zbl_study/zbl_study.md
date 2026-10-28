# ZBL 포텐셜과 switching function 정리

ZBL(Ziegler–Biersack–Littmark) 포텐셜이 무엇이고, 왜 MD/MLIP에 끼워 넣는지, 그리고 이 저장소의 `zbl/pair`(`lammps/pair_zbl_p.cpp`)처럼 switching function으로 이어 붙일 때 생기는 문제들을 정리한 노트다.

- 참고한 LAMMPS 문서: [pair zbl](https://docs.lammps.org/pair_zbl.html), [pair lj/gromacs](https://docs.lammps.org/pair_gromacs.html) (switching function), [pair tersoff/zbl](https://docs.lammps.org/pair_tersoff_zbl.html) (hybrid 예시)
- 그림과 숫자는 모두 `make_figures.py`로 다시 만들 수 있다. 출력 숫자는 `figures/numbers.txt`에 있다.
- 단위는 eV, Å, amu. ZBL 상수는 `lammps/pair_zbl_const.h`와 같다.

---

## 0. 한 장 요약

| 질문 | 짧은 답 |
|---|---|
| (1) "universal"을 DFT로 검증할 수 있나? | 있다. all-electron(또는 hard PAW) DFT로 dimer 에너지 곡선을 계산해서 비교한다. 비교해 보면 ZBL은 "평균적으로 맞는 fit"이고, pair마다 보통 수~십여 % 차이가 난다. |
| (2) 잘 맞는 구간은? | 핵간 거리가 짧아서 상호작용 에너지가 화학결합 에너지(수 eV)보다 훨씬 클 때(대략 r ≲ 1 Å, 수십 eV 이상)다. 결합 길이 근처는 틀리고, 그 사이 중간 영역은 DFT 데이터로 채워야 한다. |
| (3) 어떻게 이어 붙이나? | ① 덧셈형 switching(`pair zbl`, gromacs S(r)): inner 안쪽에서 **pair마다 상수 C만큼 shift가 생긴다.** ② 곱셈형 blending(`tersoff/zbl`의 Fermi 함수, f_c·ZBL): shift는 없지만 전이 구간에서 force에 bump가 생긴다. |
| (4) 틀린 방법이 있나? | 있다. 에너지 불연속(hard cutoff), force ≠ −∇E(가중치의 미분 항 누락), 속도·시간·이력에 의존하는 on/off는 NVE를 깨거나 보고되는 에너지를 틀리게 만든다. 너무 좁은 switching 구간은 timestep 오차를 키운다. |
| (5) shift된 ZBL이어도 괜찮은 이유? | inner 안쪽의 force는 ZBL과 완전히 같고, C는 outer 바깥의 ZBL 꼬리와 전이 구간에서 빠진 일(work)의 합이다. 단 C 자체가 공짜는 아니어서, **ML(또는 다른 포텐셜)이 그만큼을 복원해 줄 때만** 전체 에너지 곡선이 맞는다. |
| (6) reference − ZBL을 학습하는 전략은? | 표준적이고 좋은 전략이다(GAP, ACE, DeePMD `use_srtab`, MACE ZBL baseline 등). 조건은 학습과 MD에서 같은 함수를 쓰는 것, residual이 매끄러울 것(구간 폭), 짧은 거리 데이터가 있을 것. |
| (7) reference가 MLIP라서 차이가 점점 벌어지면? | switching 자체는 여전히 수학적으로 유효하다. 하지만 delta learning은 **reference를 그대로 재현**하므로 reference의 잘못된 단거리 거동까지 배운다. 신뢰할 수 없는 거리의 데이터는 빼거나 가중치를 낮추고, 그 구간은 ZBL이 맡게 해야 한다. |

---

## 1. ZBL이란

### 1.1 식

LAMMPS `pair zbl` 문서의 정의:

$$
E^{ZBL}_{ij} = \frac{1}{4\pi\varepsilon_0}\frac{Z_i Z_j e^2}{r_{ij}}\,\phi\!\left(\frac{r_{ij}}{a}\right) + S(r_{ij})
$$

$$
a = \frac{0.46850}{Z_i^{0.23}+Z_j^{0.23}}\ \text{Å}\quad(=0.8854\,a_0/(Z_i^{0.23}+Z_j^{0.23}))
$$

$$
\phi(x) = 0.18175e^{-3.19980x} + 0.50986e^{-0.94229x} + 0.28022e^{-0.40290x} + 0.02817e^{-0.20162x}
$$

- 첫 항은 **두 원자핵 사이의 쿨롱 반발** $Z_iZ_je^2/r$이다.
- $\phi(x)$는 **screening function**으로, 핵 주변 전자가 핵 전하를 가리는 효과를 나타낸다. $r\to0$이면 $\phi\to1$(맨 핵 쿨롱), $r$이 커지면 지수적으로 0이 된다.
- $a$는 screening length로, 원소쌍마다 $Z_i, Z_j$로 정해진다.
- $S(r)$는 바깥 cutoff에서 에너지·force·곡률을 0으로 만드는 switching function이다(§3).

### 1.2 왜 "universal"인가

원래 ZBL은 이온 주입·스퍼터링처럼 **keV급 이온이 고체에 충돌하는 문제**(stopping and range of ions)를 위해 만들어졌다. Ziegler, Biersack, Littmark는 여러 원소쌍에 대해 원자 전하밀도(Hartree–Fock)와 free-electron-gas 근사로 두 원자 사이 반발 포텐셜을 계산했다. 그리고 거리를 $x=r/a$로 축척하면 모든 원소쌍의 screening function이 거의 한 곡선 위에 모인다는 것을 보이고, 그 곡선을 4개의 지수 함수로 fit했다.

즉 "universal"의 의미는 다음과 같다.

> 원소쌍마다 다른 것은 $Z_iZ_j$와 축척 길이 $a$뿐이고, **모양 함수 $\phi$는 하나를 공유한다.**

ZBL이 원소와 무관하다는 뜻이 아니다. $Z$는 항상 원소별로 넣는다. 또 정확한 1차 원리 결과가 아니라 **많은 pair를 평균한 fit**이므로, 개별 pair에 대해서는 오차가 있다.

### 1.3 MD/MLIP에 왜 넣는가

- 일반 포텐셜(Tersoff, EAM, NNP, GNN-MLIP 등)은 **평형 근처 구조**에 맞춰 만들어진다. 핵간 거리가 결합 길이의 절반 이하로 가까워지는 일은 학습 데이터에 거의 없다.
- 그런 거리에서 일반 포텐셜은 반발이 너무 약하거나 오히려 인력을 보이기도 한다(NNP는 descriptor가 포화되면서 임의의 값을 extrapolation한다). 그러면 원자가 서로를 **뚫고 지나가는(unphysical fusion)** 문제가 생긴다.
- 식각·스퍼터링·이온 충돌·방사선 손상 MD에서는 수십~수천 eV의 충돌이 일어나므로 **짧은 거리 반발을 반드시 물리적으로 맞게** 기술해야 한다.
- ZBL은 이 영역에서 상당히 정확하고, 계산이 싸며, 원소쌍마다 별도 파라미터가 필요 없다. 그래서 "짧은 거리는 ZBL, 평형 근처는 원래 포텐셜"로 붙여 쓴다.

LAMMPS `tersoff/zbl` 문서의 표현: *"This screening function should be applicable to most systems. However, it is only accurate for small separations (i.e. less than 1 Angstrom)."*

---

## 2. 질문 (1): "universal"은 DFT로 검증할 수 있나?

**검증할 수 있고, 실제로 연구들이 해 왔다.** 다만 몇 가지를 알고 해야 한다.

### 2.1 어떻게 검증하나

1. 원소쌍 A–B의 **dimer**를 놓고 핵간 거리 r을 바꿔 가며 에너지를 계산한다.
   $V_{DFT}(r) = E_{AB}(r) - E_A - E_B$ (원자 에너지는 spin-polarized 바닥 상태 기준)
2. 같은 r에서 $V_{ZBL}(r)$(switching 없는 bare ZBL)과 비교한다.
3. 상대오차 $|V_{DFT}-V_{ZBL}|/V_{ZBL}$를 r 또는 V의 함수로 그린다.

Nordlund, Runeberg, Sundholm(1997)이 대표적인 연구로, HF·DFT로 계산한 dimer 반발 포텐셜과 ZBL을 비교했다. 그 결과 ZBL은 전반적으로 잘 맞지만 pair와 거리에 따라 수~십여 % 정도 차이가 났다. 그래서 이후 방사선 손상용 MLIP(예: Byggmästar et al. 2019, W GAP)에서는 bare ZBL 대신 **DFT dimer에 다시 fit한 pair별 screened Coulomb 포텐셜**을 단거리 baseline으로 쓰기도 한다.

### 2.2 DFT로 짧은 거리를 계산할 때의 함정

| 함정 | 설명 | 대처 |
|---|---|---|
| Pseudopotential/PAW core 겹침 | 두 원자 사이 거리가 core 반경 합($r_{c,A}+r_{c,B}$)보다 짧아지면 pseudopotential 근사가 깨진다. 보통 반발이 과소평가된다. | all-electron 코드(FHI-aims, PySCF, NWChem, DMol 등)를 쓰거나, VASP라면 `_h`/`_GW`/semicore 등 반경이 작은 hard PAW를 쓴다. |
| Basis 부족 | 국소 기저는 짧은 거리에서 core 영역 표현이 부족할 수 있다. | 기저를 키워 수렴을 확인한다. |
| Spin 상태 | dimer와 고립 원자의 spin 상태를 잘못 잡으면 수 eV 오차가 난다. | spin-polarized로 계산하고 여러 초기 자기모멘트를 시도한다. |
| 범함수 의존성 | 짧은 거리 에너지는 대부분 핵-핵 쿨롱과 core 전자 반발에서 오므로 XC 범함수 의존성은 **상대적으로** 작다. 다만 수십 eV 이하에서는 무시할 수 없다. | 중간 영역은 범함수 간 비교로 오차 범위를 확인한다. |

### 2.3 "dimer"로 검증해도 되나?

고체 속 충돌과 진공 dimer는 주변 원자가 다르다. 하지만 짧은 거리 반발은 두 핵과 core 전자 사이의 상호작용이 지배하므로 **주변 환경 의존성이 작다**(그래서 pair potential로 쓸 수 있다). 중간 거리(1–2 Å)에서는 환경 의존성이 커진다. 그 구간은 bulk/slab 속 충돌 snapshot DFT로 확인하는 것이 맞다. MLIP 학습 데이터가 바로 이 역할을 한다.

### 2.4 결론

- "universal"은 **검증 가능한 근사**이고, DFT(특히 all-electron) dimer 비교가 표준 검증법이다.
- 검증 결과는 보통 "짧은 거리에서는 잘 맞음, 중간 거리로 갈수록 오차 증가"다.
- 따라서 ZBL을 "정답"이 아니라 **단거리의 합리적인 baseline**으로 보고, 나머지를 DFT 데이터로 보정하는 것이 정석이다(→ 질문 6).

---

## 3. switching function의 구조 (질문 3, 5의 배경)

LAMMPS `pair zbl`의 $S(r)$은 `lj/gromacs`와 같은 함수다.

$$
S(r)=\begin{cases}
C & r<r_1\\[2pt]
\dfrac{A}{3}(r-r_1)^3+\dfrac{B}{4}(r-r_1)^4+C & r_1<r<r_c\\[2pt]
-E(r) \text{ (즉 합이 0)} & r\ge r_c
\end{cases}
$$

$$
A=\frac{-3E'(r_c)+(r_c-r_1)E''(r_c)}{(r_c-r_1)^2},\quad
B=\frac{2E'(r_c)-(r_c-r_1)E''(r_c)}{(r_c-r_1)^3}
$$

$$
C=-E(r_c)+\tfrac12(r_c-r_1)E'(r_c)-\tfrac1{12}(r_c-r_1)^2E''(r_c)
$$

경계 조건은 $S'(r_1)=S''(r_1)=0$, $S(r_c)=-E(r_c)$, $S'(r_c)=-E'(r_c)$, $S''(r_c)=-E''(r_c)$이다.

- $r_1$(inner)에서 $S$의 1·2차 미분이 0이므로, $r<r_1$에서 $S$는 **상수**일 수밖에 없다. 그 상수가 $C$다.
- $r_c$(outer)에서 $E+S$의 값·1차·2차 미분이 모두 0이다. 즉 cutoff에서 $C^2$ 연속이다.
- 따라서 $r<r_1$의 force는 bare ZBL과 **정확히 같고**, 에너지만 $C$만큼 내려가 있다.

이 저장소의 `zbl/pair`는 이 식을 그대로 쓰되, $r_1, r_c$를 **pair마다 따로** 받는다(LAMMPS 원본은 global 값 하나).

![switching](figures/switching.svg)

*그림 1. Si–Si(inner 1.598, outer 2.054 Å, 학습 스크립트 값). 왼쪽: 에너지. 덧셈형(주황)은 inner 안쪽에서 bare ZBL(파랑)보다 C = −3.56 eV 아래에 평행하게 놓인다. 곱셈형 f_c·ZBL(초록 점선)은 inner 안쪽에서 bare ZBL과 같다. 오른쪽: force. 덧셈형과 곱셈형 모두 inner 안쪽에서는 ZBL force와 같다. 곱셈형은 전이 구간에서 f_c′·E 항 때문에 ZBL보다 큰 bump가 생긴다.*

학습에 쓰는 cutoff(`generate_trainset/iter_learn/04_train/02_ZBL/get_residual_new_parallel.py`)로 계산한 $C$:

| pair | inner (Å) | outer (Å) | $E_{ZBL}(r_1)$ (eV) | $C$ (eV) | $F_{ZBL}(r_c)$ (eV/Å) |
|---|---|---|---|---|---|
| Si–Si | 1.598 | 2.054 | 7.19 | −3.56 | 5.27 |
| Si–F | 1.145 | 1.472 | 22.50 | −12.56 | 23.74 |
| Si–C | 1.208 | 1.553 | 13.80 | −7.68 | 13.79 |
| Si–H | 1.079 | 1.388 | 5.39 | −3.24 | 5.99 |
| C–F | 0.908 | 1.168 | 28.57 | −17.42 | 37.64 |
| F–F | 0.997 | 1.282 | 27.83 | −16.40 | 33.69 |
| H–H | 0.525 | 0.675 | 5.30 | −3.83 | 10.75 |

$C$는 몇 eV에서 십수 eV로, **무시할 수 있는 크기가 아니다.**

---

## 4. 질문 (2): ZBL이 잘 맞는 구간과 아닌 구간

### 4.1 세 구간

| 구간 | 대략적 범위 | 지배하는 물리 | 누가 맡아야 하나 |
|---|---|---|---|
| ① 단거리(충돌 core) | r ≲ 0.5 r_eq, 수십 eV 이상 (대부분 pair에서 r ≲ 1 Å, H 포함 pair는 더 짧음) | 핵-핵 쿨롱 + core 전자 반발. 화학결합 효과는 상대적으로 무시할 만함 | **ZBL** (또는 DFT dimer fit) |
| ② 중간 영역 | 대략 0.5–0.9 r_eq, 수 eV ~ 수십 eV | 반발과 결합 효과가 경쟁. 환경 의존성 큼 | **DFT 데이터로 학습한 포텐셜** (ZBL만으로도, 평형 포텐셜만으로도 부정확) |
| ③ 평형 근처 | r ≳ 0.9 r_eq | 화학결합 | 원래 포텐셜/MLIP. **ZBL은 꺼져 있어야 함** |

### 4.2 경계를 정하는 기준

1. **DFT dimer와의 오차**(§2): 상대오차가 허용치(예: 5–10%) 안에 드는 r을 ZBL의 "신뢰 구간"으로 잡는다.
2. **에너지 스케일**: 결합 에너지가 수 eV이므로, 상호작용 에너지가 그보다 한 자릿수 이상 큰 구간에서는 결합 효과가 오차에 미치는 비중이 작다.
3. **시뮬레이션에서 실제로 도달하는 거리**: 정면충돌의 최근접 거리 $r_{min}$은 $V(r_{min})=E_{cm}$(질량중심 에너지)로 정해진다. 아래는 ZBL로 계산한 $r_{min}$이다.

| pair | $E_{cm}$ = 10 eV | 30 eV | 100 eV | 300 eV | 1000 eV |
|---|---|---|---|---|---|
| Si–Si | 1.49 Å | 1.15 | 0.83 | 0.59 | 0.38 |
| Si–F | 1.40 | 1.06 | 0.75 | 0.52 | 0.32 |
| C–F | 1.22 | 0.90 | 0.60 | 0.40 | 0.23 |

   (실험실 에너지는 $E_{lab}=E_{cm}(M_1+M_2)/M_2$. 비정면 충돌은 $r_{min}$이 더 크다.)
   수백 eV 이온이라면 ① 구간까지 확실히 들어가고, **10–30 eV의 2차 충돌(cascade)은 대부분 ② 구간에서 일어난다.** 식각처럼 저에너지 반응이 중요한 문제에서는 ② 구간을 DFT로 잘 학습하는 것이 결과를 좌우한다.

4. **MLIP 학습 데이터 분포**: 각 pair의 최소 거리 히스토그램을 그려 본다. 데이터가 끊기는 거리보다 안쪽은 MLIP가 extrapolation하는 구간이다. 그 구간에서는 ZBL이 완전히 켜져 있어야 한다($r_1$ ≳ 데이터가 희박해지는 거리).
5. **평형 구조 보호**: $r_c$(outer)는 평형·열진동·압축 구조에서 나타나는 최소 결합 거리보다 작아야 한다. 이 저장소의 값은 결합 길이의 약 85–87%다.

---

## 5. 질문 (3): hybrid 방식으로 어떻게 이어 붙이나? constant shift가 생기나?

### 5.1 덧셈형: $E = E_{other} + \sum_{pairs}[V_{ZBL}(r)+S(r)]$

- LAMMPS `pair_style hybrid/overlay <other> zbl ...`, 이 저장소의 `hybrid/overlay nn/intel zbl/pair`가 이 방식이다.
- 다른 포텐셜은 그대로 두고, switching된 ZBL을 **더한다.**
- **constant shift가 생긴다.** inner 안쪽에서 pair 에너지가 $V_{ZBL}+C$다(§3).
- 장점: inner 안쪽의 force가 ZBL과 정확히 같고, 기존 포텐셜 코드를 건드리지 않는다.
- 단점: 다른 포텐셜이 짧은 거리에서 내는 에너지·force가 **그대로 남는다.** 그래서 ZBL을 더해도 그 오차가 같이 들어간다. NNP라면 descriptor 포화로 짧은 거리 force가 작게 나오는 편이지만 보장은 없다.

### 5.2 곱셈형(보간): $V = (1-f)\,V_{ZBL} + f\,V_{other}$

- LAMMPS `tersoff/zbl`:

$$
V_{ij}=(1-f_F(r_{ij}))V^{ZBL}(r_{ij})+f_F(r_{ij})V^{Tersoff}(r_{ij}),\qquad
f_F(r)=\frac{1}{1+e^{-A_F(r-r_C)}}
$$

  $A_F$는 전이의 날카로움, $r_C$는 사실상 ZBL cutoff다.
- **constant shift가 없다.** $r\ll r_C$이면 $V\to V_{ZBL}$, $r\gg r_C$이면 $V\to V_{other}$로 각각의 원래 값에 수렴한다.
- 단점: force에 $f_F'(r)\,(V_{other}-V_{ZBL})$ 항이 생긴다. 두 포텐셜의 차이가 크면 전이 구간에 force의 bump나 dip이 생긴다. $A_F$를 너무 크게 하면 이 항이 커지고, 너무 작게 하면 두 포텐셜이 넓게 섞인다. Fermi 함수는 정확히 0/1이 되지 않으므로 아주 작은 섞임은 항상 남는다.
- MLIP에서는 NN 에너지가 원자별 합이라 pair마다 곱하기 어렵다. 그래서 보통 **ZBL에만 envelope를 곱하는** 변형을 쓴다: $E = E_{ML} + \sum f_c(r)V_{ZBL}(r)$, $f_c$는 $r<r_1$에서 1, $r>r_c$에서 0인 매끄러운 다항식. 이것도 shift가 없지만 전이 구간 force에 $f_c'V_{ZBL}$ bump가 생긴다(그림 1의 초록 점선).

### 5.3 비교

| | 덧셈형 S(r) (`pair zbl`) | 곱셈형 f_c·ZBL | Fermi 보간 (`tersoff/zbl`) |
|---|---|---|---|
| r < inner 에너지 | ZBL + **C** | ZBL | ≈ ZBL |
| r < inner force | ZBL | ZBL | ≈ ZBL (+작은 섞임) |
| 전이 구간 force | ZBL보다 **작음** (일이 빠짐 → C) | ZBL보다 **큼** (f_c′·E bump) | f′·(V_other − V_ZBL) 항 |
| 연속성 | outer에서 $C^2$ | 다항식 차수에 따라 $C^1$/$C^2$ | $C^\infty$ (단 정확히 0이 되지 않음) |
| 다른 포텐셜 단거리 값 | 그대로 더해짐 | 그대로 더해짐 | (1−f)로 억제됨 |

**요약하면:** constant shift는 "덧셈형 + inner 안쪽 force 보존"을 택했을 때 생기는 수학적 귀결이다. 곱셈형을 쓰면 shift는 없지만 그 대신 전이 구간 force가 ZBL보다 커진다. 어느 쪽도 공짜는 아니다. 전이 구간에서 "빠진 일"이 shift로 나타나거나, "더해진 일"이 bump로 나타날 뿐이다.

---

## 6. 질문 (4): 잘못된 이어붙이기가 있나? (NVE 보존)

### 6.1 NVE 보존의 조건

Velocity Verlet처럼 symplectic한 적분기로 에너지가 (오차 범위 내에서) 보존되려면 다음이 필요하다.

1. **force가 정확히 −∇E**(보고하는 그 에너지의 gradient)일 것. 그렇지 않으면 보고되는 에너지가 보존되지 않는다.
2. **E가 위치만의 함수**일 것. 속도·시간·이력(어떤 원자가 "에너지가 높았는지" 등)에 의존하면 계가 conservative하지 않다.
3. **E가 연속($C^0$)**이고, 실용적으로는 **force도 연속($C^1$)**일 것. 불연속이 있으면 cutoff를 넘는 순간 에너지가 튀거나 적분 오차가 크게 생긴다.
4. **dt가 force 변화 속도에 비해 충분히 작을 것.** 위 조건을 다 만족해도 dt가 크면 에너지가 출렁이고 drift한다.

### 6.2 1D 충돌 실험

Si–Si 정면충돌($E_{cm}$ = 30 eV, dt = 1 fs, velocity Verlet)을 다섯 가지 결합 방식으로 돌려서 총에너지 $E_{tot}$의 변화를 봤다.

![nve](figures/nve.svg)

| 방식 | 충돌 중 최대 \|ΔE\| | 충돌 후 ΔE | 판정 |
|---|---|---|---|
| (a) ZBL + S(r) (gromacs) | 0.44 eV | +0.011 eV | 정상. 충돌 순간의 출렁임은 dt 오차이고 돌아온다 |
| (b) hard cutoff (S 없음) | 2.29 eV | +0.023 eV | **불량.** cutoff를 넘을 때 $V_{ZBL}(r_c)$만큼 에너지가 점프한다. 고립 dimer라 들어갈 때와 나올 때가 상쇄됐을 뿐이고, 다체계에서는 이런 점프가 쌓여 drift가 된다 |
| (c) Fermi 보간, gradient 완전 | 0.52 eV | +0.038 eV | 정상 |
| (d) Fermi 보간, $f'$ 항 누락 | **7.31 eV** | −0.011 eV | **불량.** force가 보고하는 에너지의 gradient가 아니다. 이 1D pair 예에서는 force가 여전히 r만의 함수라서 "다른 어떤 포텐셜"에 대해서는 conservative하고, 그래서 끝에서 값이 돌아왔다. 가중치가 배위수 같은 다체 변수에 의존하면 진짜로 비보존이 된다 |
| (e) 속도에 따라 ZBL on/off | 6.26 eV | **−3.01 eV** | **불량.** 포텐셜이 속도에 의존하므로 에너지가 영구적으로 사라진다 |

### 6.3 "틀린" 방법 정리

- **Hard cutoff / 불연속 switching**: 에너지가 점프한다. `pair_style zbl`이 switching을 내장하고 `pair_modify shift`를 지원하지 않는 이유가 이것이다(문서: *"the ZBL interaction is already smoothed to 0.0 at the cutoff"*).
- **force만 switching하고 에너지는 그대로 보고**(또는 반대): force와 에너지가 불일치하므로 보고되는 에너지가 보존되지 않는다.
- **곱셈형 가중치의 미분을 빠뜨림**: $f$가 r에 의존하면 $f'(V_{other}-V_{ZBL})$ 항이 필요하다. 가중치가 이웃 원자 위치에 의존(배위수 기반 switching 등)하면 그 이웃에 대한 force 항도 모두 넣어야 한다.
- **운동에너지·시간·이력 기반 on/off**: "빠른 원자에만 ZBL 적용", "처음 N step만 ZBL" 같은 방식이다. NVE가 깨진다.
- **학습 때와 MD 때 다른 ZBL**: NVE는 보존되지만 **물리가 틀린다.** 학습할 때 뺀 $C$·switching 형태와 MD에서 더하는 것이 다르면 NN이 보정하는 양과 실제 더해지는 양이 어긋난다.
- **`hybrid`(overlay가 아님)로 pair를 배타적으로 할당**: MLIP는 다체 포텐셜이라 pair 단위로 끄고 켤 수 없다. ZBL은 `hybrid/overlay`로 더해야 한다.

### 6.4 형식적으로는 맞지만 위험한 경우: 너무 좁은 switching 구간

S(r)는 구간 폭 $r_c-r_1$이 좁을수록 전이 구간의 force 기울기(곡률)가 커진다. N–N 정면충돌에서 한 번 충돌한 뒤 남는 $|\Delta E|$:

| N–N 설정 | $E_{cm}$ | dt = 1.0 fs | 0.5 fs | 0.2 fs | 0.1 fs |
|---|---|---|---|---|---|
| bare ZBL | 30 eV | 0.19 | 0.043 | 0.007 | 0.002 |
| 넓은 구간 0.78/1.00 Å | 30 eV | 0.57 | 0.48 | 0.016 | 0.003 |
| 좁은 구간 0.95/1.00 Å | 30 eV | 0.57 | 2.01 | **0.84** | 0.098 |
| bare ZBL | 100 eV | 11.2 | 0.22 | 0.023 | 0.006 |
| 넓은 구간 | 100 eV | 26.4 | 6.0 | 0.0005 | 0.008 |
| 좁은 구간 | 100 eV | 19.9 | 9.4 | **1.47** | 0.064 |

(단위 eV)

- 같은 dt에서 좁은 구간이 오차가 훨씬 크다(30 eV, 0.2 fs: 0.016 → 0.84 eV).
- 수십 eV 이상의 충돌에는 구간 폭과 무관하게 **dt ≲ 0.1–0.2 fs**가 필요하다. LAMMPS에서는 `fix dt/reset`으로 최대 변위를 제한하는 가변 dt를 흔히 쓴다.
- MLIP와 결합할 때는 좁은 구간이 residual도 날카롭게 만든다(질문 6).

---

## 7. 질문 (5): constant shift가 있는 ZBL이어도 괜찮은 이유

### 7.1 괜찮은 부분

- **inner 안쪽의 force는 bare ZBL과 똑같다.** 충돌 core에서 원자를 튕겨 내는 힘은 정확하다.
- 이 shift는 계 전체의 상수가 아니라 **pair 함수의 일부**다. $V_{ZBL}(r)+S(r)$는 모든 r에서 $C^2$로 매끄러운 하나의 pair potential이다. pair가 inner를 넘나들 때 에너지 점프가 없으므로 NVE와 아무 충돌이 없다.
- 이 함수에서 유도한 force가 dynamics를 정하고, 보고되는 에너지와도 일치한다.

### 7.2 공짜는 아닌 부분

$$
C=-\Big[\,V_{ZBL}(r_c)+\int_{r_1}^{r_c}\big(F_{ZBL}(r)-F_{sw}(r)\big)\,dr\Big]
$$

($F=-dE/dr$, $F_{sw}$는 ZBL+S의 force.) 즉 $|C|$는 **outer 바깥으로 잘려 나간 ZBL 꼬리 에너지 $V_{ZBL}(r_c)$**와 **전이 구간에서 force를 줄이면서 빠진 일(work)**을 합한 값이다. Si–Si 예에서는 $V_{ZBL}(r_c)$ ≈ 2.10 eV, 빠진 일 ≈ 1.46 eV로 합이 3.56 eV다. 그래서 ZBL+S만 단독으로 쓰면, 무한히 먼 곳에서 inner 안쪽까지 들어오는 데 필요한 에너지가 bare ZBL보다 $|C|$만큼 작다. 충돌의 최근접 거리가 달라진다:

| pair | $E_{cm}$ | $r_{min}$ (bare ZBL) | $r_{min}$ (ZBL+S) |
|---|---|---|---|
| Si–Si | 10 eV | 1.49 Å | 1.39 Å |
| Si–F | 10 eV | 1.40 | 1.14 |
| C–F | 10 eV | 1.22 | 0.92 |
| C–F | 100 eV | 0.60 | 0.57 |
| C–F | 1000 eV | 0.23 | 0.23 |

- 고에너지(≳ 수백 eV)에서는 $|C|$가 상대적으로 작아서 차이가 무시할 만하다.
- **저에너지(수~수십 eV)에서는 차이가 크다.** 이 영역이 식각 반응에서 중요하다.

### 7.3 그래서 왜 괜찮은가 → 다른 항이 $C$를 복원하기 때문

- **Delta learning(이 저장소 방식)**: NN이 $E_{ref}-E_{ZBL+S}$를 학습하므로, NN 출력에 "가까운 pair당 대략 $-C$"가 들어간다. 전체 모델 $E_{NN}+E_{ZBL+S}\approx E_{ref}$가 되어 $C$는 상쇄된다. 남는 것은 NN이 그 residual을 얼마나 잘 표현하느냐(§8)다.
- **Tersoff 등에 ZBL을 그냥 overlay**(재fit 없이): 아무도 $C$를 복원하지 않는다. 저에너지 충돌의 반발 장벽이 $|C|$만큼 낮아진다. 이런 경우에는 shift 없는 곱셈형을 쓰거나, 포텐셜을 재fit하는 것이 맞다.

**결론:** "collision force만 필요하니 shift는 상관없다"는 inner 안쪽 force에 대해서는 맞다. 하지만 충돌 궤적(최근접 거리, 산란각, 에너지 전달)은 force를 경로를 따라 적분한 에너지로 정해진다. 그래서 **전체 에너지 곡선이 reference를 따라가야** 한다. delta learning을 하면 이 조건이 자동으로 만족된다. 이것이 shift된 ZBL을 써도 괜찮은 진짜 이유다.

---

## 8. 질문 (6): reference − ZBL을 학습하는 전략은 괜찮은가

### 8.1 표준적인 전략이다

$$
E_{model}=E_{ZBL+S}+E_{ML},\qquad E_{ML}\ \text{은}\ E_{ref}-E_{ZBL+S}\ \text{(force, stress도 동일)를 학습}
$$

- GAP(Byggmästar et al. 등 방사선 손상 포텐셜): pair repulsion baseline 위에 GAP를 학습한다.
- DeePMD-kit: `use_srtab`으로 단거리 pair 테이블을 넣고, 그 위에 NN을 학습한다.
- ACE, MACE, NequIP 계열에도 ZBL/core repulsion baseline 옵션이 있다.

### 8.2 장점

- NN이 수십~수백 eV의 가파른 반발을 직접 배우지 않아도 된다. residual은 크기가 훨씬 작고 완만하다.
- 데이터가 없는 아주 짧은 거리에서도 ZBL이 지배하므로 원자가 서로 뚫고 지나가지 않는다.
- 이 저장소는 학습할 때 residual을 **같은 LAMMPS `zbl/pair`로** 계산해서 빼므로(E, F, S 모두) 학습과 MD 사이의 일관성이 확보되어 있다.

### 8.3 성공 조건과 체크리스트

1. **일관성**: MD의 `pair_coeff` inner/outer가 학습 때 쓴 `cutoff_dict`와 같아야 한다. 이 저장소에서는 `etc/nnp_neb/*_footer.in`의 값(SiSi 1.94/2.05 등)이 학습 스크립트(SiSi 1.598/2.054)와 다르다. 같은 potential이라면 맞춰야 한다.
2. **residual의 매끄러움**: 전이 구간 폭이 넓을수록 좋다. 좁으면 residual에 ZBL force만큼의 급격한 계단이 생기고, 매끄러운 descriptor를 쓰는 NN은 이를 잘 못 맞춘다. 결합 길이의 약 0.78 → 1.0배 정도인 학습 스크립트 쪽 비율이 무난하다.
3. **단거리 데이터**: 중간 영역(②)의 DFT 데이터(dimer scan, 압축 구조, 충돌 snapshot)가 있어야 NN이 ZBL과 DFT의 차이를 배운다.
4. **inner 안쪽에서 NN의 거동**: inner 안쪽의 residual은 "pair당 $-C$ + (DFT−ZBL 오차)"로 대략 일정하다. NN은 그 근처를 extrapolation한다. dimer/trimer scan으로 $F_{NN}$이 $F_{ZBL}$에 비해 작은지 확인한다.
5. **loss weighting**: 충돌 구조의 force는 크므로 force loss가 그 구조에 끌려가지 않도록 가중치나 force clipping을 점검한다.

---

## 9. 질문 (7): reference가 DFT가 아니라 MLIP이고, 차이가 점점 벌어지면?

### 9.1 먼저 구분할 것

- **constant shift $C$는 "ZBL+S vs bare ZBL"의 관계**다. reference와는 무관하게 S(r)만으로 정해진다.
- **"reference − ZBL"은 처음부터 상수가 아니다.** DFT라도 r에 따라 변한다. 다만 신뢰할 만한 reference라면 짧은 거리로 갈수록 ZBL에 **수렴**(상대오차 감소)하고, MLIP처럼 단거리 데이터가 없는 reference는 **발산**(softening, 가짜 인력)하는 경향이 있다.

### 9.2 그림으로 보면

![reference](figures/reference_residual.svg)

*그림 3. 가상의 두 reference. A(DFT형)는 짧은 거리에서 ZBL보다 약 5% 높게 따라가고, residual(오른쪽)은 inner 안쪽에서 거의 일정하다. B(MLIP형, softening)는 1.6 Å 아래에서 ZBL보다 점점 낮아지고, residual이 짧은 거리에서 −30 eV 이상으로 계속 커진다.*

### 9.3 switching 전략은 여전히 유효한가?

- **수학적으로는 유효하다.** ZBL+S는 reference와 무관하게 매끄럽고 보존적인 pair potential이다.
- **문제는 delta learning이 reference를 그대로 재현한다는 점이다.** reference B의 데이터를 inner 안쪽까지 학습하면 NN이 "−30 eV짜리 residual"을 배운다. 그러면 전체 모델 = ZBL + S + NN ≈ reference B가 되어 **ZBL을 넣은 의미가 사라진다.** reference가 틀린 곳에서는 모델도 틀린다.
- 게다가 residual이 크고 가파르면 NN이 잘 맞추지도 못한다. 그 결과 ZBL도 reference도 아닌 어중간한 곡선이 나온다.

### 9.4 어떻게 해야 하나

1. **reference의 신뢰 거리 $r_{trust}$를 정한다.** 소수의 DFT dimer/충돌 계산과 비교해서 reference가 어디서부터 틀어지는지 확인한다.
2. **$r_{trust}$보다 짧은 pair가 있는 구조는 학습에서 빼거나 가중치를 크게 낮춘다.** 그 구간은 ZBL이 맡는다.
3. **$r_1$(inner)을 $r_{trust}$ 근처 또는 바깥에 둔다.** reference가 틀리기 시작하는 거리에서는 ZBL이 이미 완전히 켜져 있어야 한다.
4. 가능하면 **단거리만이라도 DFT 데이터를 섞는다.** 소량의 dimer/trimer scan만으로도 효과가 크다.
5. 검증: dimer scan과 저에너지 충돌 궤적을 DFT와 비교한다. 최근접 거리, 에너지 손실, 반사/흡착 여부를 본다.

**결론:** 차이가 점점 벌어지는 reference라도 switching function 자체는 문제가 없다. 하지만 **"벌어지는 구간의 reference 데이터를 학습하지 않는다"**는 조건을 지켜야 전략이 성립한다. delta learning은 "reference가 옳다"는 가정 위에서만 옳은 결과를 준다.

---

## 10. 이 저장소에 적용한 정리

- `zbl/pair` = LAMMPS `pair zbl`과 같은 식 + pair별 inner/outer. pair별 cutoff는 물리적으로 문제없다. LAMMPS가 global 값을 쓰는 것은 구현상의 단순화다.
- 학습(`get_residual_new_parallel.py`)과 MD가 같은 바이너리와 같은 함수를 쓰므로 $C$는 NN이 흡수한다. 따라서 shift는 문제가 아니다.
- 확인할 것:
  - MD 입력의 `pair_coeff` inner/outer = 학습 때의 `cutoff_dict`인지(`etc/nnp_neb`는 다름).
  - 전이 구간을 충분히 넓게 유지할 것(현재 학습 값 비율 ≈ 0.78은 무난).
  - 충돌 MD의 dt(≲ 0.1–0.2 fs 또는 `fix dt/reset`).
  - dimer scan으로 NN+ZBL과 DFT를 비교(특히 ② 구간).
- 코드 메모: `write_restart`가 inner/outer를 저장하지 않으므로 restart 파일에서 이어 돌리면 값이 틀어진다.

---

## 참고 문헌

- J. F. Ziegler, J. P. Biersack, U. Littmark, *The Stopping and Range of Ions in Matter*, Vol. 1, Pergamon (1985).
- K. Nordlund, N. Runeberg, D. Sundholm, "Repulsive interatomic potentials calculated using Hartree-Fock and density-functional theory methods", *Nucl. Instrum. Methods B* **132**, 45 (1997).
- J. Tersoff, "New empirical approach for the structure and energy of covalent systems", *Phys. Rev. B* **37**, 6991 (1988).
- J. Byggmästar, A. Hamedani, K. Nordlund, F. Djurabekova, "Machine-learning interatomic potential for radiation damage and defects in tungsten", *Phys. Rev. B* **100**, 144105 (2019).
- E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, Springer (2006). symplectic 적분기와 에너지 보존.
- LAMMPS 문서: `pair_style zbl`, `pair_style lj/gromacs`, `pair_style tersoff/zbl`.
