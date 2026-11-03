# EOGProject

EOG(안구전도, electrooculography) 신호를 실시간으로 전처리하고, 1D CNN으로 안구 운동을 분류하는 프로젝트입니다.

- **전처리 (C)** — 기준선 드리프트 제거 + Savitzky-Golay 평활. 샘플이 하나씩 들어오는 실시간 처리 구조입니다.
- **분류 (MATLAB → C)** — MATLAB Deep Learning Toolbox로 학습한 1D CNN이 750ms 윈도우를 **U / D / L / R / idle / blink** 6개 클래스로 분류합니다. 학습한 가중치는 C로 내보내서 MATLAB 없이 추론할 수 있습니다.
- **성능** — 6명 데이터로 피험자 단위 교차검증(leave-one-subject-out)을 했고, 처음 보는 사람 기준 **평균 93.9%** 입니다.

```
원본 .mat ──► convert_dataset.py ──► .eog ──► filter_dataset (C)
                                                  │
                                    ┌─────────────┴─────────────┐
                                    ▼                           ▼
                              _filtered.csv                   .filt
                                    │                           │
                              plot_eog.m (그래프)     eog_build_dataset.m
                                                                │
                                                      eog_train_cnn.m / eog_cross_validate.m
                                                                │
                                                      eog_export_cnn.m
                                                                │
                                                      src/eog_cnn.c (C 추론)
```

---

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `src/eog_epoch.*` | 에폭 파이프라인: 100샘플 구간을 10샘플마다 표준화하고 SG 필터를 적용 |
| `src/eog_normalize.*` | O(1) 롤링 평균·표준편차. 10초 스케일 창에 사용 |
| `src/eog_dataset.*` | `.eog` 파일 로더 |
| `src/eog_cnn.*` | CNN 추론 (동적 메모리 할당 없음) |
| `src/eog_cnn_weights.h` | 학습된 가중치. **자동 생성 파일이므로 직접 수정하지 마세요** |
| `tools/convert_dataset.py` | MATLAB v5 `.mat` → `.eog` 변환 (Python 표준 라이브러리만 사용) |
| `tools/filter_dataset.c` | 녹화 데이터 전체를 필터링해 CSV/바이너리로 출력 |
| `tools/plot_eog.m` | 원본과 필터 결과를 겹쳐 그리기 (좌우 y축 분리) |
| `matlab/` | 데이터셋 구성, CNN 학습, 교차검증, C 내보내기 |
| `test/` | C 단위 테스트, MATLAB 출력과 일치하는지 검증 |
| `examples/pipeline_demo.c` | 합성 신호로 돌려보는 파이프라인 예제 |
| `Savitzky-Golay-Filter/` | SG 필터 라이브러리 (submodule, [원본](https://github.com/Tugbars/Savitzky-Golay-Filter)을 fork) |
| `data/` | 생성물 (변환 데이터, 필터 출력, 학습 결과). **저장소에는 포함되지 않습니다** |

---

## 필요한 것

| 도구 | 용도 | 비고 |
|---|---|---|
| git | 저장소 받기 | |
| MSYS2 UCRT64의 gcc, cmake, ninja | C 빌드 | `pacman -S mingw-w64-ucrt-x86_64-{gcc,cmake,ninja}` |
| MSYS2 UCRT64의 gdb | 디버깅 (선택) | `pacman -S mingw-w64-ucrt-x86_64-gdb` |
| Python 3 | `.mat` 변환 | 추가 패키지 필요 없음 |
| MATLAB + Deep Learning Toolbox | CNN 학습·내보내기, 그래프 | R2026a에서 확인 |

C 추론만 쓰신다면 MATLAB은 필요 없습니다. 가중치 헤더가 저장소에 들어 있습니다.

`C:\msys64\ucrt64\bin`이 PATH에 있어야 합니다. MSYS2의 gcc는 이 폴더에 있는 DLL이 없으면 실행되지 않습니다.

---

## 시작하기

### 1. 받기

submodule까지 함께 받아야 합니다.

```bash
git clone --recursive https://github.com/bravesean05/EOGProject.git
```

이미 받으셨는데 `Savitzky-Golay-Filter/` 폴더가 비어 있다면 이렇게 받습니다.

```bash
git submodule update --init --recursive
```

### 2. 빌드와 테스트

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
```

테스트 7개가 모두 통과하면 정상입니다. `Release`를 빼고 빌드하면 최적화가 꺼져서 CNN 추론이 6~9배 느려집니다.

VS Code로 열 때는 이 저장소의 루트가 아니라 **`Savitzky-Golay-Filter/` 폴더를 따로 열면** 그쪽 설정(`.vscode/`)이 적용됩니다.

---

## 데이터셋 처리

### 데이터셋

[Barbara et al.의 EOG 데이터셋](https://doi.org/10.1016/j.bspc.2019.101738)을 사용합니다. **저장소에는 포함되어 있지 않으니** 따로 받으셔야 합니다.

- 피험자 6명(S1~S6), 1인당 300 trial, trial당 4초
- 256Hz, 수평·수직 2채널. 녹화할 때 이미 0~30Hz 대역통과와 50Hz 노치 필터를 거쳤습니다
- trial 구성: forward saccade(0~1초) → return saccade(1~2초) → blink(2~4초)

아래 명령에서 `<DATASET>`은 `S1`~`S6` 폴더가 들어 있는 경로로 바꿔 쓰세요.

### 3. 변환

```bash
python tools/convert_dataset.py "<DATASET>" data/converted
```

### 4. 필터링

```bash
for s in S1 S2 S3 S4 S5 S6; do ./build/filter_dataset data/converted/$s.eog -o data/${s}_filtered.csv -b data/$s.filt; done
```

- `-o` : 그래프용 CSV
- `-b` : MATLAB 학습용 바이너리 (CSV의 1/10 크기)

필터링이 끝나면 레이블별 요약이 출력됩니다.

### 5. 그래프 (MATLAB)

MATLAB의 현재 폴더를 저장소 루트로 두고 실행하세요.

```matlab
addpath('tools');
plot_eog('data/S1_filtered.csv', 'TimeRange', [100 108])
```

원본(왼쪽 축)과 필터 출력(오른쪽 축)이 겹쳐 그려지고, 배경색으로 정답 레이블이 표시됩니다. 31만 행을 읽는 데 시간이 걸리므로 여러 구간을 볼 때는 테이블을 받아서 다시 넘기세요.

```matlab
T = plot_eog('data/S1_filtered.csv');
plot_eog(T, 'TimeRange', [540 548], 'Channel', 'h')
```

---

## CNN 학습 (MATLAB)

모든 명령은 **저장소 루트**에서 실행하세요. 출력 파일 경로가 이 위치를 기준으로 잡혀 있습니다.

```matlab
addpath('matlab');

% 6. 레이블이 붙은 윈도우 만들기
DS = eog_build_dataset('DatasetDir', '<DATASET>', 'Save', 'data/eog_windows.mat');

% 7. 교차검증: 6명을 한 명씩 테스트로 돌림 (약 20분)
CV = eog_cross_validate(DS, 'Save', 'data/eog_cv.mat');

% 8. 최종 모델: 6명 전체로 학습
R = eog_train_cnn(DS, 'TestSubject', '', 'ValSubject', '', 'MaxEpochs', 40, ...
                  'Save', 'data/eog_cnn_final.mat');

% 9. C로 내보내기
eog_export_cnn(R, DS);
```

9번은 `src/eog_cnn_weights.h`와 `test/eog_cnn_vectors.h`를 다시 만듭니다. 그다음 C를 다시 빌드하고, 전체 윈도우에서 MATLAB 출력과 일치하는지 확인하세요.

```bash
cmake --build build && ./build/test_eog_cnn data/eog_cnn_check.bin
```

### 레이블을 만든 방법

원래 데이터셋에는 forward / return / blink 구분만 있어서, 6개 클래스는 직접 규칙을 정해 만들었습니다.

| 클래스 | 만든 방법 |
|---|---|
| **U D L R** | `TargetGA.mat`의 목표 좌표에서 **우세 축**으로 판정합니다 (\|h\| ≥ \|v\|면 좌/우, 아니면 상/하). 목표가 화면의 무작위 지점이라서, **45° 대각선 ±10° 안의 saccade는 제외**했습니다. return saccade는 forward와 반대 방향입니다. |
| **blink** | blink 레이블 2초 구간 안에서, 수직 채널의 50ms 포락선이 1.0을 넘는 짧은 버스트(중앙값 375ms) |
| **idle** | 같은 구간에서 버스트로부터 0.15초 이상 떨어진 조용한 부분. 원래 데이터셋에는 idle 레이블이 없습니다. |

윈도우는 구간 시작부터 192샘플(750ms)입니다. 데이터는 반드시 **피험자 단위로 나눠야** 합니다. 같은 saccade에서 나온 윈도우들은 거의 똑같아서, 무작위로 나누면 실제보다 점수가 높게 나옵니다.

---

## C에서 추론하기

```c
#include "eog_cnn.h"

EogCnnWork work;                      /* 작업 메모리 48KB, 재사용 가능 */
float probs[EOG_CNN_NUM_CLASSES];

/* window: 192샘플 × 2채널, 시간 순서로 h0 v0 h1 v1 ... */
int cls = eog_cnn_classify(window, probs, &work);
printf("%s (%.1f%%)\n", eog_cnn_class_name(cls), 100.0f * probs[cls]);
```

입력은 반드시 **`eog_epoch`을 거친 신호**여야 합니다. 원본을 넣으면 학습 때와 스케일이 달라서 결과를 믿을 수 없습니다.

---

## 결과

### 전처리 설정 (256Hz 기준)

| 항목 | 값 |
|---|---|
| 에폭 | 100샘플 (390.6ms) |
| stride | 10샘플 (39.1ms) |
| SG 커널 | 11샘플 (43.0ms), 2차 |
| 스케일 창 | 2560샘플 (10초) |
| 필터 지연 | 19.5~54.7ms |

표준화할 때 에폭마다 그 구간의 표준편차로 나누면 조용한 구간의 노이즈가 커져서 saccade와 구분이 안 됩니다(분리도 1.04배). 그래서 10초 창의 표준편차로 나누고, 블록 경계에서 기준선을 이어주도록 바꿨습니다. 그 결과 S1 수평 채널에서 saccade와 blink 구간의 출력 크기 차이가 **6.3배**가 됐습니다.

### 교차검증 (leave-one-subject-out)

| 테스트 피험자 | S1 | S2 | S3 | S4 | S5 | S6 | **평균** |
|---|---|---|---|---|---|---|---|
| 정확도 | 96.4% | 93.6% | 96.2% | 88.1% | 94.9% | 94.4% | **93.9%** |

| 클래스 | blink | idle | L | R | U | D |
|---|---|---|---|---|---|---|
| 평균 recall | 99.4% | 94.6% | 92.4% | 91.0% | 88.8% | 83.9% |

### C 추론

- MATLAB과 비교: 6,333개 윈도우 **전부 같은 클래스**, 확률 차이 최대 1.07×10⁻⁶
- 속도: 윈도우당 0.7~1.9ms (Release 빌드, 개발 PC 기준이며 측정 시점의 부하에 따라 다름). 분류 간격 39ms의 2~5%
- 메모리: 작업 버퍼 48KB, 가중치 36,262개

---

## 한계

- **레이블은 규칙으로 만든 것입니다.** 정확도는 사람이 직접 표시한 정답이 아니라 위 규칙과 얼마나 일치하는지를 잰 값입니다.
- **대각선 움직임은 평가하지 않았습니다.** 학습에서 제외했기 때문에 대각선 입력에 어떻게 반응할지는 알 수 없습니다.
- **U/D 표본이 적습니다** (각 328개, L/R의 약 1/3). 그래서 U/D 성능이 가장 낮고 피험자에 따라 흔들립니다 (D: 70~95%).
- **실시간으로 돌려보지 않았습니다.** 학습과 평가 모두 움직임 시작 시점에 맞춰 자른 윈도우로 했습니다. 실제 장치처럼 윈도우를 밀어가며 분류하면 오검출이 생길 수 있고, 판정 시점은 움직임 시작 후 약 620~660ms로 예상됩니다(측정값이 아니라 추정입니다).
- **같은 실험실, 같은 장비에서 녹화한 6명의 데이터입니다.** 다른 장비나 일상 환경에서도 같은 성능이 나온다는 보장은 없습니다.

---

## 참고

- 데이터셋: N. Barbara, T. A. Camilleri, K. P. Camilleri, "A comparison of EOG baseline drift mitigation techniques," *Biomedical Signal Processing and Control*, vol. 57, 2020. [doi:10.1016/j.bspc.2019.101738](https://doi.org/10.1016/j.bspc.2019.101738)
- Savitzky-Golay 필터: [Tugbars/Savitzky-Golay-Filter](https://github.com/Tugbars/Savitzky-Golay-Filter) (MIT License)
