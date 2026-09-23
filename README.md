# samsungkeyboard — 손 스캔 랜드마크 정합 및 엄지 길이 예측

후임자용 인수인계 문서입니다. 상세 수치와 분석은 아래 두 문서에 있고, 이 파일은 "무엇을 왜 했고, 어디까지 왔고, 다음에 무엇을 하면 되는지"만 담습니다.
- [align_readme.md](align_readme.md) — lnd → obj 정합 방법과 결과
- [align2_and_modeling_readme.md](align2_and_modeling_readme.md) — 정합 수정 경위, 랜드마크 구조, 엄지 길이 예측 모델 결과와 한계 분석

## 1. 목적
손 3D 스캔 메쉬(`obj/`)만으로 비공개 SW가 찍은 랜드마크 중 **P1(엄지 끝)·P15(엄지 기저)** 를 재현하고, 둘 사이 **xy 평면 거리 = 엄지손가락 직선길이(Size Korea 393 항목)** 를 0.5mm 오차 이내로 예측하는 것. 즉 비공개 SW의 블랙박스 재현입니다.

## 2. 데이터 (15명, 파일명 = HUMAN_ID + "G")
| 폴더 | 내용 | 비고 |
|---|---|---|
| `obj/` | rapidform 출력 손 메쉬, mm, 바닥 z≈0, 팔 쪽 y≈−100 절단 | **원본. 수정 금지** |
| `lnd/` | 비공개 SW가 찍은 랜드마크 28개 (`번호 x y z`), mm | **원본. 수정 금지**. obj와 좌표계가 다름 |
| `aligned/` | lnd를 obj 좌표계로 옮긴 결과 (`_aligned.lnd`, `_landmarks.ply`, `_combined.obj`, `_transform.txt`, `summary.csv`) | `python align_landmarks.py`로 재생성 |
| `thumb_length.csv` | 사람별 엄지 길이(P1–P15 xy)와 기준값 비교 — 15/15 일치 | |
| `PLY/`, `STL/` | 5명분, obj와 동일한 메쉬(색·텍스처 없음) | 정보 없음, git 제외 |

랜드마크 구조(15명 전부 0.000mm로 검증): 1~6, 15~27은 SW가 3D로 찍은 점. 7~14는 각 손가락 끝(P2~5)→밑(P16~19) 직선의 1/3·2/3 지점(xy 계산, z=0), 28은 P26·P27 중점(z=1). lnd 좌표계는 손목 중점(P28)→가운데 손가락 끝(P3)이 +y.

기준값(엄지 길이 15명분)은 `align_landmarks.py`의 `REFERENCE_THUMB_MM`에 있습니다.

## 3. 환경·실행
Python 3.12, numpy 2.3, scipy 1.18, scikit-learn 1.9, pandas. GPU·torch 없음(필요 없음).
```
python align_landmarks.py                 # 정합 → aligned/, thumb_length.csv   (약 5분)
python thumb_model/prep.py                # 정점별 특징 캐시 → thumb_model/cache/ (약 1분, git 제외라 처음 한 번 실행)
python thumb_model/train_eval.py          # 12명 LOO + 3명 테스트 + model_hgb.pkl (약 17분)
python thumb_model/train_eval.py --stage1 mlp   # MLP 비교 (약 90분, 결과 동일)
python thumb_model/predict.py obj/20_F_0179G.obj   # 추론: P1·P15 좌표 + 길이, thumb_model/predictions/*.ply
```

## 4. 현재 상태 (2026-09-23 기준)
| 항목 | 상태 |
|---|---|
| lnd → obj 정합 | **완료.** 15명 모두 실측 19점 평균 표면 거리 0.17~0.28mm |
| 엄지 길이 검증 | **완료.** 15명 모두 기준값과 ±0.05mm |
| P1(엄지 끝) 재현 | **성공.** 메쉬 엄지 끝점 기반, 축 방향 오차 표준편차 0.6mm |
| P15(엄지 기저) 재현 | **미해결.** 형상만으로 약 2mm가 한계 |
| 엄지 길이 예측 | 15명 held-out MAE **2.10mm** (목표 0.5mm 미달) |

## 5. 반드시 알아야 할 판단과 교훈
1. **정합은 z축 회전 + xyz 이동만 허용.** 두 데이터 모두 바닥이 z=0이고, 이렇게 해야 xy 길이(엄지 길이)가 보존됩니다. 3D 회전을 허용하면 길이가 최대 0.64mm 왜곡됩니다.
2. **정합에는 실측 3D 점 19개만 사용**(`NON_3D_LANDMARKS` = 7~14, 28 제외). 채워 넣은 z가 정합을 위로 끌어올렸습니다.
3. **trimmed ICP를 쓰면 안 됩니다.** 손가락은 튜브라 축 방향으로 미끄러져도 표면 거리가 안 변하고, 버린 점이 하필 밀어줄 점이라 5~9mm 덜 밀린 채 수렴했습니다. 지금은 점 전부 사용 + 손 축 방향 다중 시작(−6~+16mm)으로 해결. 정합 품질은 표면 거리만 보지 말고 `summary.csv`의 `tip_apex_gap_mm`(P1과 메쉬 엄지 끝점의 축 방향 간격)로 확인하세요.
4. **P15 오차는 학습 알고리즘 문제가 아닙니다.** 부스팅·MLP·랜덤 포레스트·SVR·아틀라스 ICP·전역 회귀·블렌드 모두 같은 사람들(0097, 1113, 1573)에서 같은 방향으로 틀립니다. P15는 유리판에 눌린 평평한 손바닥 위라 형상 단서가 없고, 보이는 기하 대비 위치가 사람마다 ±5mm 다릅니다. 이 편차는 12명으로 학습할 수 없습니다.
5. **테스트 3명 수치는 어느 3명이냐에 따라 0.2~4.6mm로 흔들립니다.** 성능은 15명 전체 held-out MAE(2.1mm)로 보고하세요. 12:3 반복 분할은 오차를 줄이지 못합니다.
6. 테스트 3명(`TEST_SUBJECTS` = 0179, 2635, 1116)은 고정입니다. 바꾸면 `results_*`도 다시 생성해야 합니다.

## 6. 다음 단계 (막힌 지점을 뚫으려면)
P15를 0.5mm 수준으로 재현하려면 SW가 P15를 잡을 때 본 것과 같은 입력이 필요합니다. 우선순위 순:
1. **스캐너/SW의 다른 출력물 확보**: 텍스처가 붙은 메쉬, 손바닥 사진, 원본 고해상도 스캔, 2D 랜드마크 파일. 색이나 텍스처가 있으면 주름을 직접 찾을 수 있습니다. (obj/PLY/STL은 전부 같은 메쉬라 소용없음)
2. **표본 추가**: 같은 SW로 찍은 손이 수십 명 이상이면 형상으로 설명되는 부분만이라도 더 배울 수 있습니다. `lnd/`, `obj/`에 같은 이름으로 넣고 `REFERENCE_THUMB_MM`에 기준값을 추가한 뒤 위 실행 순서를 그대로 돌리면 됩니다.
3. **P15 정의 변경**(최후): 엄지–손바닥 접합부처럼 메쉬에서 재현 가능한 점으로 바꾸면 모델은 정밀해지지만 기준값과 사람마다 수 mm 달라집니다. 이 길이의 용도가 주름 기준을 요구하지 않을 때만 고려.

## 7. 저장소 관리
- git 제외: `thumb_model/cache/`(prep.py로 재생성), `thumb_model/predictions/`, `thumb_model/model_mlp.pkl`, `PLY/`, `STL/`, `__pycache__/`.
- `thumb_model/model_hgb.pkl`(29MB)은 `predict.py`가 쓰는 최종 모델이라 포함.
- 결과물을 `obj/`, `lnd/`에 쓰지 마세요. 예측 PLY는 `thumb_model/predictions/`로 갑니다.
