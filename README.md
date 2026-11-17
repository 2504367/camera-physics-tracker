# 스마트폰 카메라 기반 물리 실험 데이터 자동 분석 툴

스마트폰으로 찍은 공 낙하·포물선 운동 영상을 분석해서, 프레임별 위치·속도·가속도와 중력가속도 g를 자동으로 계산하는 툴입니다.

## 주제

- 물리 원리: 2차원 평면 운동학(등가속도 운동), 에너지 보존
- 알고리즘: 색상 기반 객체 추적(HSV 마스킹, 윤곽선 검출), 픽셀→미터 보정, 곡선 맞춤
- 개발 방식: 바이브 코딩

## 진행 상황 (1일차)

- [x] HSV 색상 마스킹과 윤곽선 검출로 공 추적
- [x] 프레임별 픽셀 좌표 CSV 저장
- [x] 실제 영상 없이 시험할 수 있는 테스트 영상 생성기
- [ ] 픽셀 → 미터 보정
- [ ] 속도·가속도·g 계산과 그래프
- [ ] 사용 편의 기능 (클릭 설정, 구간 자동 선택)

## 설치

```
pip install -r requirements.txt
```

## 사용법

```
python make_test_video.py
python ball_tracker.py test_drop.mp4 --color orange --save-mask
```

`test_drop_result/track_px.csv`에 프레임별 공 중심 픽셀 좌표가 저장됩니다. `--save-mask`를 붙이면 공이 제대로 잡히는지 확인할 수 있는 마스크 이미지도 저장됩니다.

색상 프리셋: orange, yellow, green, blue, red, pink. 직접 지정하려면 `--hsv h1,s1,v1,h2,s2,v2`.
