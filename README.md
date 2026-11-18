# 스마트폰 카메라 기반 물리 실험 데이터 자동 분석 툴

스마트폰으로 찍은 공 낙하·포물선 운동 영상을 분석해서, 프레임별 위치·속도·가속도와 중력가속도 g를 자동으로 계산하는 툴입니다.

## 주제

- 물리 원리: 2차원 평면 운동학(등가속도 운동), 에너지 보존
- 알고리즘: 색상 기반 객체 추적(HSV 마스킹, 윤곽선 검출), 픽셀→미터 보정, 곡선 맞춤
- 개발 방식: 바이브 코딩

## 진행 상황 (2일차)

- [x] HSV 색상 마스킹과 윤곽선 검출로 공 추적
- [x] 프레임별 픽셀 좌표 CSV 저장
- [x] 실제 영상 없이 시험할 수 있는 테스트 영상 생성기
- [x] 기준 자 양 끝 좌표와 실제 길이로 픽셀 → 미터 보정
- [x] 2차 곡선 맞춤으로 g 추정 (표준오차, 95% 구간, R²)
- [x] 국소 2차 회귀로 프레임별 속도·가속도 계산
- [x] 결과 CSV, 그래프(위치·속도·가속도), 요약 JSON
- [ ] 사용 편의 기능 (클릭 설정, 구간 자동 선택)

## 설치

```
pip install -r requirements.txt
```

## 사용법

```
python make_test_video.py
python ball_tracker.py test_drop.mp4 --color orange --scale-points 40,70,40,370 --scale-length 1.0 --start 0.26 --end 0.75
```

- `--scale-points x1,y1,x2,y2`: 영상 속 기준 자 양 끝의 픽셀 좌표
- `--scale-length`: 기준 자의 실제 길이 (m)
- `--start`, `--end`: 공을 놓은 직후 ~ 바닥에 닿기 직전 구간 (초). 지정하지 않으면 정지 구간과 튕긴 구간이 섞여서 결과가 틀어집니다.
- `--fps`: 슬로모션 영상은 실제 촬영 fps를 지정해야 합니다.

`test_drop_result/`에 track.csv, analysis.png, summary.json이 저장됩니다.

색상 프리셋: orange, yellow, green, blue, red, pink. 직접 지정하려면 `--hsv h1,s1,v1,h2,s2,v2`.
