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
python thumb_model/rule_pipeline.py                 # 특허 규칙 파이프라인(학습 없음) → rule_eval.csv, rule_landmarks.csv (약 3분)
python thumb_model/eval_all_landmarks.py            # 28점 전체 오차, 학습 모델·기준선 15-fold LOO (약 45분)
python thumb_model/landmark_report.py               # 접근별 28점 오차 보고서 → thumb_model/report/ (그림 4개, HTML, 요약 CSV)
```

접근별 28점 오차 비교 자료(팀 공유용): `thumb_model/report/landmark_report.html` — 28점 평균 3D 오차 규칙 3.40 / 학습 2.75 / 기준선 6.34mm. 상세는 `align2_and_modeling_readme.md`의 "접근별 전체 랜드마크 오차" 절.

## 4. 현재 상태 (2026-09-24 기준)
**최종 판정: 정답 P15를 재현하는 방법(규칙·수정 규칙·학습 모델)은 찾지 못했습니다.** 모든 접근이 P15 표준편차 약 2.4mm에서 멈추며, 엄지 길이 최선은 MAE 2.1mm(목표 0.5mm 미달). 접근별 수치 표는 `align2_and_modeling_readme.md` 맨 위 "최종 판정" 절.

| 항목 | 상태 |
|---|---|
| lnd → obj 정합 | **완료.** 15명 모두 실측 19점 평균 표면 거리 0.17~0.28mm |
| 엄지 길이 검증 | **완료.** 15명 모두 기준값과 ±0.05mm |
| P1(엄지 끝) 재현 | **성공.** 메쉬 엄지 끝점 기반, 축 방향 오차 표준편차 0.6mm |
| P15(엄지 기저) 재현 | **미해결.** 형상만으로 약 2mm가 한계 |
| 엄지 길이 예측 | 15명 held-out MAE **2.10mm** (목표 0.5mm 미달) |
| 특허 규칙 기반 재현 (hy1) | SW는 특허 KR 10-1217207 계열로 확인(구조 일치). 기저점 규칙은 문구와 달라 P15 ±2.4mm에서 멈춤. `thumb_model/rule_pipeline.py` |

## 5. 반드시 알아야 할 판단과 교훈
1. **정합은 z축 회전 + xyz 이동만 허용.** 두 데이터 모두 바닥이 z=0이고, 이렇게 해야 xy 길이(엄지 길이)가 보존됩니다. 3D 회전을 허용하면 길이가 최대 0.64mm 왜곡됩니다.
2. **정합에는 실측 3D 점 19개만 사용**(`NON_3D_LANDMARKS` = 7~14, 28 제외). 채워 넣은 z가 정합을 위로 끌어올렸습니다.
3. **trimmed ICP를 쓰면 안 됩니다.** 손가락은 튜브라 축 방향으로 미끄러져도 표면 거리가 안 변하고, 버린 점이 하필 밀어줄 점이라 5~9mm 덜 밀린 채 수렴했습니다. 지금은 점 전부 사용 + 손 축 방향 다중 시작(−6~+16mm)으로 해결. 정합 품질은 표면 거리만 보지 말고 `summary.csv`의 `tip_apex_gap_mm`(P1과 메쉬 엄지 끝점의 축 방향 간격)로 확인하세요.
4. **P15 오차는 학습 알고리즘 문제가 아닙니다.** 부스팅·MLP·랜덤 포레스트·SVR·아틀라스 ICP·전역 회귀·블렌드 모두 같은 사람들(0097, 1113, 1573)에서 같은 방향으로 틀립니다. P15는 유리판에 눌린 평평한 손바닥 위라 형상 단서가 없고, 보이는 기하 대비 위치가 사람마다 ±5mm 다릅니다. 이 편차는 12명으로 학습할 수 없습니다.
5. **테스트 3명 수치는 어느 3명이냐에 따라 0.2~4.6mm로 흔들립니다.** 성능은 15명 전체 held-out MAE(2.1mm)로 보고하세요. 12:3 반복 분할은 오차를 줄이지 못합니다.
6. 테스트 3명(`TEST_SUBJECTS` = 0179, 2635, 1116)은 고정입니다. 바꾸면 `results_*`도 다시 생성해야 합니다.

## 6. 다음 단계 (막힌 지점을 뚫으려면)
목표는 **SW 출력값(= 기준값)의 재현**입니다. 기준값 "실제(mm)"는 SW의 P1–P15 거리와 0.05mm 이내로 같으므로 SW 출력이고, 캘리퍼 실측값은 없습니다(확인됨). 따라서 SW 자체의 정확도는 알 수 없고, P15를 0.5mm 수준으로 재현하려면 SW가 P15를 잡을 때 쓴 규칙이나 입력이 필요합니다. 우선순위 순:
1. **SW의 기저점(첫째마디중심점) 정의 문의**: SW가 특허 KR 10-1217207(디엔엠에프티, 손 자동계측 방법) 계열임이 데이터로 확인됐습니다(손목중심 규칙, 실루엣 web 점, 손가락 축 구조가 일치). 그러나 기저점 규칙은 공개 문구("web 점을 축에 수직 투영")와 달라 P15가 ±2.4mm 남습니다. 특허가 공개돼 있으므로 공급사에 기저점 정의(문구·상수)만 확인하면 규칙으로 바로 재현할 수 있습니다. 상세: `align2_and_modeling_readme.md`의 "특허 규칙 기반 검토".
2. **스캐너/SW의 다른 출력물 확보**: 절단 전 원본 스캔(특허 기준 해상도 0.3mm, 팔 포함 — 특허의 손목 규칙은 팔 부분이 있어야 동작), 텍스처가 붙은 메쉬, 손바닥 사진, 2D 랜드마크 파일. (obj/PLY/STL은 전부 같은 메쉬라 소용없음)
3. **표본 추가**: 같은 SW로 찍은 손이 수십 명 이상이면 형상으로 설명되는 부분만이라도 더 배울 수 있습니다. `lnd/`, `obj/`에 같은 이름으로 넣고 `REFERENCE_THUMB_MM`에 기준값을 추가한 뒤 위 실행 순서를 그대로 돌리면 됩니다.
4. **Size Korea 393 항목의 공식 정의 확인**: P15가 주름점인지 관절점인지 문서로 확정되면 어떤 출력물이 결정적인지 알 수 있습니다.

**검토 후 기각 — 다른 랜드마크로 P15 예측**: 정답 랜드마크 18개를 전부 알아도 선형 회귀 P15 오차 ≥ 2.7mm, 비선형(RF·SVR·GP·KNN 등, 21개 해부학 특징) 최선 1.6mm(최대 4mm). P15는 저장된 다른 점들의 함수가 아닙니다(70,192개 기하 구성 전수 탐색 최선 3.4mm).

**검토 후 기각 — P15 정의 변경**: 메쉬에서 규칙으로 계산되는 다른 점(엄지–손바닥 접합부, 엄지–검지 오목점 투영, 손바닥면 들림 최대점)을 P15로 새로 정의하는 방안을 검토했으나, 이 길이의 용도가 Size Korea 기준값과의 비교·대체라서 맞지 않습니다. 세 후보 모두 정답 P15와 축 방향으로 ±5~8mm(표준편차) 어긋나 기준값 일치도가 현재 모델(±2.5mm)보다 나쁩니다. 상세는 `align2_and_modeling_readme.md`의 "P15 정의 변경 검토" 절. 다시 시도하지 마세요.

## 7. 저장소 관리
- git 제외: `thumb_model/cache/`(prep.py로 재생성), `thumb_model/predictions/`, `thumb_model/model_mlp.pkl`, `PLY/`, `STL/`, `__pycache__/`.
- `thumb_model/model_hgb.pkl`(29MB)은 `predict.py`가 쓰는 최종 모델이라 포함.
- 결과물을 `obj/`, `lnd/`에 쓰지 마세요. 예측 PLY는 `thumb_model/predictions/`로 갑니다.
