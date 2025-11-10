# 음악 지리적 원산지 데이터 시각화

UCI Geographical Origin of Music Dataset을 활용한 다양한 시각화 프로젝트입니다.

## 데이터셋
- `default_features_1059_tracks.csv`: 기본 음악 특징 데이터 (1059개 트랙)
- `default_plus_chromatic_features_1059_tracks.csv`: 크로마틱 특징 포함 데이터

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

```bash
python visualize_music_data.py
```

## 생성되는 시각화

1. **Pairplot** (`outputs/pairplot.png`)
   - 주요 음악 특징 간의 상관관계를 국가별로 시각화

2. **Violinplot** (`outputs/violinplot.png`)
   - 국가별 음악 특징 분포를 바이올린 플롯으로 표시

3. **Scatterplot** (`outputs/scatterplot.png`)
   - 여러 특징 간의 산점도 및 지리적 위치 시각화

4. **World Map** (`outputs/world_map.png`)
   - Geopandas를 이용한 세계 지도 시각화
   - 트랙 위치, 국가별 샘플 수, 특징 분포 등을 지도에 표시

5. **Country Cluster Map** (`outputs/country_cluster_map.png`)
   - K-means 클러스터링을 이용한 국가별 음악적 특성 분류

## 출력 디렉토리
모든 시각화 결과는 `outputs/` 디렉토리에 저장됩니다.

