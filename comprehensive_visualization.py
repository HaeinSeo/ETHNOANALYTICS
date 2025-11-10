"""
종합 시각화 및 XAI 분석 스크립트
- 5가지 이상의 다양한 시각화 기법
- 5가지 이상의 XAI 기법 적용
- GeoPandas choropleth 시각화 포함
- 인터랙티브 대시보드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import shap
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import importlib
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')


def ensure_package(package_name: str) -> bool:
    """Ensure that a Python package is available; attempt installation if missing."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        print(f"패키지 '{package_name}'가 설치되어 있지 않습니다. 설치를 시도합니다...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"  ✓ '{package_name}' 설치 완료")
            return True
        except Exception as install_err:
            print(f"  ⚠️ '{package_name}' 설치 실패: {install_err}")
            return False


if not ensure_package("geopandas"):
    raise ImportError("GeoPandas는 필수 패키지입니다. 설치 후 다시 실행해주세요.")

HAS_MAPCLASSIFY = ensure_package("mapclassify")
if not HAS_MAPCLASSIFY:
    print("경고: 'mapclassify' 패키지가 설치되어 있지 않습니다. 일부 GeoPandas 시각화에서 분위수 범례를 비활성화합니다.")

import geopandas as gpd
from shapely.geometry import Point
if HAS_MAPCLASSIFY:
    import mapclassify  # noqa: F401

# 한글 폰트 설정 (Windows)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 폰트가 없을 경우 기본 폰트 사용
    pass

print("=" * 80)
print("종합 시각화 및 XAI 분석 시작")
print("=" * 80)

# 1. 데이터 로드 및 전처리
print("\n[1단계] 데이터 로드 및 전처리...")
df = pd.read_csv('data/default_features_1059_tracks.csv', header=0)
print(f"데이터 형태: {df.shape}")
print(f"컬럼 수: {len(df.columns)}")

# 마지막 두 컬럼이 좌표인지 확인
coords_cols = df.columns[-2:].tolist()
print(f"좌표 컬럼 (추정): {coords_cols}")

# 피처와 좌표 분리
feature_cols = df.columns[:-2].tolist()
X = df[feature_cols].copy()
coords = df[coords_cols].copy()

print(f"피처 수: {len(feature_cols)}")
print(f"좌표 범위: lat [{coords.iloc[:, 0].min():.2f}, {coords.iloc[:, 0].max():.2f}], "
      f"lon [{coords.iloc[:, 1].min():.2f}, {coords.iloc[:, 1].max():.2f}]")

# 타겟 생성 (클러스터링 기반)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
y_cluster = kmeans.fit_predict(X_scaled)

# 추가 타겟: 첫 번째 주성분 기반 (회귀용)
pca_temp = PCA(n_components=1)
y_regression = pca_temp.fit_transform(X_scaled).ravel()

# 피처 그룹화 (역할군/주제별)
# 음악 피처로 가정: spectral, temporal, rhythmic, harmonic, timbral 등
n_features = len(feature_cols)
feature_groups = {
    'Spectral': feature_cols[:n_features//5],
    'Temporal': feature_cols[n_features//5:2*n_features//5],
    'Rhythmic': feature_cols[2*n_features//5:3*n_features//5],
    'Harmonic': feature_cols[3*n_features//5:4*n_features//5],
    'Timbral': feature_cols[4*n_features//5:]
}

print(f"\n피처 그룹화:")
for group, cols in feature_groups.items():
    print(f"  {group}: {len(cols)}개 피처")

# 저장 디렉토리 생성
import os
os.makedirs('visualizations', exist_ok=True)
os.makedirs('xai_results', exist_ok=True)

print("\n[2단계] 시각화 생성 시작...")

# ============================================================================
# 시각화 1: 상관관계 Heatmap
# ============================================================================
print("\n[시각화 1] 상관관계 Heatmap 생성...")
fig, ax = plt.subplots(figsize=(16, 14))
corr_matrix = X.corr()
# 샘플링하여 계산 속도 향상 (필요시)
if len(corr_matrix) > 50:
    sample_idx = np.random.choice(len(corr_matrix), 50, replace=False)
    corr_sample = corr_matrix.iloc[sample_idx, sample_idx]
else:
    corr_sample = corr_matrix

sns.heatmap(corr_sample, cmap='coolwarm', center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax, 
            fmt='.2f', annot=False)
ax.set_title('Feature Correlation Heatmap\n(상관관계 히트맵)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: visualizations/01_correlation_heatmap.png")

# ============================================================================
# 시각화 2: PCA 2D 및 3D 시각화
# ============================================================================
print("\n[시각화 2] PCA 시각화 생성...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 2D PCA
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_cluster, 
                    cmap='viridis', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
ax.set_title('PCA 2D Visualization\n(주성분 분석 2D 시각화)', 
             fontsize=16, fontweight='bold', pad=20)
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/02_pca_2d.png', dpi=300, bbox_inches='tight')
plt.close()

# 3D PCA (Plotly)
fig = go.Figure(data=[go.Scatter3d(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    z=X_pca[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=y_cluster,
        colorscale='Viridis',
        opacity=0.7,
        line=dict(width=0.5, color='black')
    ),
    text=[f'Sample {i}' for i in range(len(X_pca))],
    hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
)])
fig.update_layout(
    title=dict(
        text='PCA 3D Visualization (주성분 분석 3D 시각화)',
        font=dict(size=16)
    ),
    scene=dict(
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
        zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
    ),
    width=900,
    height=800
)
fig.write_html('visualizations/02_pca_3d.html')
print("  ✓ 저장 완료: visualizations/02_pca_2d.png, 02_pca_3d.html")

# ============================================================================
# 시각화 3: 역할군/주제별 피처 그룹화 시각화
# ============================================================================
print("\n[시각화 3] 그룹별 피처 시각화 생성...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (group_name, group_cols) in enumerate(feature_groups.items()):
    if idx >= 6:
        break
    group_data = X[group_cols]
    
    # 그룹별 평균값을 클러스터별로 비교
    group_means_by_cluster = []
    for cluster_id in range(3):
        cluster_mask = y_cluster == cluster_id
        group_means_by_cluster.append(group_data[cluster_mask].mean().values)
    
    group_means_by_cluster = np.array(group_means_by_cluster)
    
    ax = axes[idx]
    x_pos = np.arange(len(group_cols))
    width = 0.25
    
    for i, cluster_id in enumerate(range(3)):
        ax.bar(x_pos + i*width, group_means_by_cluster[i], width, 
              label=f'Cluster {cluster_id}', alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=10)
    ax.set_ylabel('Mean Value', fontsize=10)
    ax.set_title(f'{group_name} Features by Cluster\n({group_name} 피처 그룹별 클러스터 비교)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'F{i}' for i in range(len(group_cols))], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# 빈 서브플롯 제거
if len(feature_groups) < 6:
    for idx in range(len(feature_groups), 6):
        axes[idx].remove()

plt.suptitle('Feature Groups Visualization by Cluster\n(역할군별 피처 그룹화 시각화)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visualizations/03_feature_groups.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: visualizations/03_feature_groups.png")

# ============================================================================
# 시각화 4: 모델 기반 중요도 시각화
# ============================================================================
print("\n[시각화 4] 모델 기반 중요도 시각화 생성...")
# 랜덤 포레스트 모델 학습
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cluster, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 피처 중요도 추출
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# 상위 20개 피처 시각화
top_n = 20
top_features = feature_importance.head(top_n)

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(range(len(top_features)), top_features['importance'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([f'Feature {f}' for f in top_features['feature']])
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title(f'Top {top_n} Feature Importance (Random Forest)\n(상위 {top_n}개 피처 중요도)', 
             fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('visualizations/04_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 중요도 기반 필터링된 피처 시각화 (상위 10개)
important_features = feature_importance.head(10)['feature'].tolist()
X_important = X[important_features]

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(X_important.T, cmap='RdYlBu_r', center=0, ax=ax, 
            cbar_kws={"label": "Feature Value"}, xticklabels=False)
ax.set_title('Top 10 Important Features Heatmap\n(상위 10개 중요 피처 히트맵)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Features', fontsize=12)
ax.set_xlabel('Samples', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/04_important_features_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: visualizations/04_feature_importance.png, 04_important_features_heatmap.png")

# ============================================================================
# 시각화 5: GeoPandas Choropleth 스타일 시각화
# ============================================================================
print("\n[시각화 5] GeoPandas Choropleth 시각화 생성...")
# 좌표 데이터로 GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(coords.iloc[:, 1], coords.iloc[:, 0])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# 중요 피처 값으로 choropleth 생성
gdf['important_feature'] = X_important.mean(axis=1)

# 좌표 범위 확인 및 그리드 생성
bounds = gdf.total_bounds
print(f"좌표 범위: {bounds}")

# 그리드 기반 집계 (choropleth 효과를 위한)
from shapely.geometry import box
import math

# 그리드 크기 계산
grid_size = 5  # 도 단위
x_min, y_min, x_max, y_max = bounds

# 그리드 생성
grid_cells = []
for x in np.arange(x_min, x_max, grid_size):
    for y in np.arange(y_min, y_max, grid_size):
        grid_cells.append(box(x, y, x + grid_size, y + grid_size))

grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:4326')

# 그리드에 포인트 집계
gdf_grid = gpd.sjoin(gdf, grid_gdf, how='left', predicate='within')
aggregated = gdf_grid.groupby('index_right')['important_feature'].mean().reset_index()
grid_gdf = grid_gdf.merge(aggregated, left_index=True, right_on='index_right', how='left')
grid_gdf['important_feature'] = grid_gdf['important_feature'].fillna(0)

# Choropleth 시각화
fig, ax = plt.subplots(figsize=(16, 12))
grid_gdf.plot(column='important_feature', cmap='YlOrRd', 
              legend=True, ax=ax, edgecolor='black', linewidth=0.5,
              legend_kwds={'label': 'Feature Value', 
                          'shrink': 0.8,
                          'fmt': '{:.2f}'})
gdf.plot(ax=ax, color='black', markersize=2, alpha=0.5, label='Data Points')
ax.set_title('Geospatial Feature Distribution (Choropleth)\n(지리적 피처 분포 - 등치선도)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/05_geopandas_choropleth.png', dpi=300, bbox_inches='tight')
plt.close()

# GeoPandas 스타일 범례 포함
if HAS_MAPCLASSIFY:
    fig, ax = plt.subplots(figsize=(16, 12))
    grid_gdf.plot(column='important_feature', scheme='QUANTILES', k=5,
                  cmap='YlOrRd', legend=True, ax=ax, edgecolor='black', linewidth=0.5,
                  legend_kwds={'loc': 'center left', 
                              'bbox_to_anchor': (1, 0.5),
                              'fmt': '{:.2f}',
                              'interval': True})
    gdf.plot(ax=ax, color='black', markersize=2, alpha=0.3)
    ax.set_title('Geospatial Feature Distribution with Quantiles\n(지리적 피처 분포 - 분위수 범례)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/05_geopandas_choropleth_quantiles.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("  ⚠️ 'mapclassify' 미설치로 분위수 기반 Choropleth를 건너뜁니다.")

# ============================================================================
# 시각화 5-1: 세계 지도 기반 시각화
# ============================================================================
print("\n[시각화 5-추가] 세계 지도 시각화 생성...")
world_map_created = False
try:
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    fig, ax = plt.subplots(figsize=(18, 10))
    world.plot(ax=ax, color='lightgray', edgecolor='white')
    scatter = gdf.plot(
        ax=ax,
        column='important_feature',
        cmap='YlOrRd',
        markersize=20,
        legend=True,
        alpha=0.7,
        legend_kwds={'label': 'Average Important Feature Value', 'shrink': 0.7}
    )
    ax.set_title('Global Distribution of Important Feature Values\n(세계 지도 기반 중요 피처 분포)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/05_world_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 저장 완료: visualizations/05_world_map.png")
    world_map_created = True
except Exception as e:
    print(f"  ⚠️ 세계 지도 시각화 생성 중 오류 발생: {e}")

if HAS_MAPCLASSIFY:
    print("  ✓ 저장 완료: visualizations/05_geopandas_choropleth.png, 05_geopandas_choropleth_quantiles.png")
else:
    print("  ✓ 저장 완료: visualizations/05_geopandas_choropleth.png")

# ============================================================================
# 시각화 6: 인터랙티브 대시보드 (Plotly)
# ============================================================================
print("\n[시각화 6] 인터랙티브 대시보드 생성...")
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('PCA 2D Scatter', 'Feature Importance Top 10', 
                   'Geospatial Distribution', 'Cluster Distribution'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"type": "scattermapbox"}, {"secondary_y": False}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. PCA 2D
fig.add_trace(
    go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        mode='markers',
        marker=dict(size=5, color=y_cluster, colorscale='Viridis', showscale=True),
        text=[f'Sample {i}' for i in range(len(X_pca))],
        name='PCA'
    ),
    row=1, col=1
)

# 2. Feature Importance
top_10 = feature_importance.head(10)
fig.add_trace(
    go.Bar(
        x=top_10['importance'],
        y=[f'F{f}' for f in top_10['feature']],
        orientation='h',
        marker=dict(color=top_10['importance'], colorscale='Viridis'),
        name='Importance'
    ),
    row=1, col=2
)

# 3. Geospatial
fig.add_trace(
    go.Scattermapbox(
        lat=coords.iloc[:, 0],
        lon=coords.iloc[:, 1],
        mode='markers',
        marker=dict(size=5, color=X_important.mean(axis=1), 
                   colorscale='YlOrRd', showscale=True,
                   colorbar=dict(x=1.05, len=0.4)),
        text=[f'Sample {i}' for i in range(len(coords))],
        name='Location'
    ),
    row=2, col=1
)

# 4. Cluster Distribution
cluster_counts = pd.Series(y_cluster).value_counts().sort_index()
fig.add_trace(
    go.Bar(
        x=[f'Cluster {i}' for i in cluster_counts.index],
        y=cluster_counts.values,
        marker=dict(color=cluster_counts.index, colorscale='Viridis'),
        name='Cluster Count'
    ),
    row=2, col=2
)

# 업데이트 레이아웃
fig.update_layout(
    title_text="Interactive Dashboard - Comprehensive Data Analysis<br>(인터랙티브 대시보드 - 종합 데이터 분석)",
    title_font_size=18,
    height=900,
    showlegend=False,
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=coords.iloc[:, 0].mean(), lon=coords.iloc[:, 1].mean()),
        zoom=2
    )
)

fig.update_xaxes(title_text="PC1", row=1, col=1)
fig.update_yaxes(title_text="PC2", row=1, col=1)
fig.update_xaxes(title_text="Importance Score", row=1, col=2)
fig.update_xaxes(title_text="Cluster", row=2, col=2)
fig.update_yaxes(title_text="Count", row=2, col=2)

fig.write_html('visualizations/06_interactive_dashboard.html')
print("  ✓ 저장 완료: visualizations/06_interactive_dashboard.html")

print("\n[3단계] XAI 분석 시작...")

# ============================================================================
# XAI 1: SHAP Summary Plot
# ============================================================================
print("\n[XAI 1] SHAP Summary Plot 생성...")
# 샘플링하여 계산 속도 향상
sample_size = min(100, len(X_test))
feature_aliases = [f'F{i}' for i in range(X_test.shape[1])]
X_shap_input = X_test[:sample_size]
X_shap_df = pd.DataFrame(X_shap_input, columns=feature_aliases)

explainer = shap.TreeExplainer(rf_model)
shap_values_raw = explainer.shap_values(X_shap_input)


def _normalize_shap_values(values):
    """Return list of shap value matrices with shape (n_samples, n_features)."""
    if isinstance(values, list):
        return values
    values = np.array(values)
    if values.ndim == 3:
        return [values[:, i, :] for i in range(values.shape[1])]
    return [values]


def _select_expected_value(expected, class_index=0):
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected_array = np.array(expected).reshape(-1)
        idx = min(class_index, len(expected_array) - 1)
        return expected_array[idx]
    return expected


shap_values_list = _normalize_shap_values(shap_values_raw)
primary_class_index = 0
shap_values_primary = shap_values_list[primary_class_index]

if len(shap_values_list) > 1:
    shap_values_stack = np.stack([np.abs(vals) for vals in shap_values_list], axis=0)
    shap_values_for_bar = shap_values_stack.mean(axis=0)
else:
    shap_values_for_bar = shap_values_primary

# Summary plot (bar)
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_values_for_bar,
    X_shap_input,
    feature_names=feature_aliases,
    plot_type='bar',
    show=False,
    max_display=20
)
plt.title('SHAP Summary Plot (Bar)\n(SHAP 요약 플롯 - 막대)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('xai_results/01_shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# Summary plot (dot) - 첫 번째 클래스 기준
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_values_primary,
    X_shap_input,
    feature_names=feature_aliases,
    show=False,
    max_display=20
)
plt.title('SHAP Summary Plot (Dot)\n(SHAP 요약 플롯 - 점)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('xai_results/01_shap_summary_dot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: xai_results/01_shap_summary_bar.png, 01_shap_summary_dot.png")

# ============================================================================
# XAI 2: SHAP Waterfall Plot
# ============================================================================
print("\n[XAI 2] SHAP Waterfall Plot 생성...")
# 첫 번째 샘플에 대한 waterfall plot
expected_value_primary = _select_expected_value(explainer.expected_value, primary_class_index)
waterfall_explanation = shap.Explanation(
    values=shap_values_primary[0],
    base_values=expected_value_primary,
    data=X_shap_df.iloc[0].values,
    feature_names=feature_aliases
)

fig, ax = plt.subplots(figsize=(12, 8))
shap.waterfall_plot(waterfall_explanation, show=False, max_display=15)
plt.title('SHAP Waterfall Plot\n(SHAP 워터폴 플롯)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('xai_results/02_shap_waterfall.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: xai_results/02_shap_waterfall.png")

# ============================================================================
# XAI 3: SHAP Force Plot (HTML)
# ============================================================================
print("\n[XAI 3] SHAP Force Plot 생성...")
force_fig = shap.force_plot(
    expected_value_primary,
    shap_values_primary[0],
    X_shap_df.iloc[0].values,
    feature_names=feature_aliases,
    matplotlib=True,
    show=False
)
force_fig.savefig('xai_results/03_shap_force.png', dpi=300, bbox_inches='tight')
plt.close(force_fig)

# HTML 버전
force_plot_html = shap.force_plot(
    expected_value_primary,
    shap_values_primary[:10],
    X_shap_df.iloc[:10],
    feature_names=feature_aliases,
    show=False
)
shap.save_html('xai_results/03_shap_force.html', force_plot_html)
print("  ✓ 저장 완료: xai_results/03_shap_force.png, 03_shap_force.html")

# ============================================================================
# XAI 4: SHAP Dependence Plot
# ============================================================================
print("\n[XAI 4] SHAP Dependence Plot 생성...")
# 가장 중요한 피처에 대한 dependence plot
top_feature_idx = feature_importance.iloc[0]['feature']
if isinstance(top_feature_idx, str):
    top_feature_idx = int(top_feature_idx)
else:
    top_feature_idx = int(top_feature_idx)
target_feature_name = feature_aliases[top_feature_idx]

fig, ax = plt.subplots(figsize=(10, 8))
shap.dependence_plot(
    target_feature_name,
    shap_values_primary,
    X_shap_df,
    feature_names=feature_aliases,
    show=False
)
plt.title(f"SHAP Dependence Plot - Feature {target_feature_name}\n(SHAP 의존성 플롯)", 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('xai_results/04_shap_dependence.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: xai_results/04_shap_dependence.png")

# ============================================================================
# XAI 5: LIME
# ============================================================================
print("\n[XAI 5] LIME 분석 생성...")
# LIME explainer 생성
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=[f'F{i}' for i in range(X_train.shape[1])],
    class_names=[f'Cluster {i}' for i in range(3)],
    mode='classification'
)

# 첫 번째 샘플 설명
exp = lime_explainer.explain_instance(
    X_test[0],
    rf_model.predict_proba,
    num_features=15
)

# LIME 시각화
fig = exp.as_pyplot_figure()
plt.title('LIME Explanation\n(LIME 설명)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('xai_results/05_lime_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

# LIME HTML 저장
exp.save_to_file('xai_results/05_lime_explanation.html')
print("  ✓ 저장 완료: xai_results/05_lime_explanation.png, 05_lime_explanation.html")

# ============================================================================
# XAI 6: Partial Dependence Plot (PDP)
# ============================================================================
print("\n[XAI 6] Partial Dependence Plot 생성...")
# 회귀 모델 학습 (PDP용)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_regression[:len(X_train)])

# 상위 3개 피처에 대한 PDP
top_3_features = feature_importance.head(3)['feature'].tolist()
top_3_indices = [int(f) for f in top_3_features]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (feature_idx, ax) in enumerate(zip(top_3_indices, axes)):
    # PDP 계산
    feature_range = np.linspace(X.iloc[:, feature_idx].min(), 
                               X.iloc[:, feature_idx].max(), 50)
    pdp_values = []
    
    for val in feature_range:
        X_temp = X_shap_input.copy()
        X_temp[:, feature_idx] = val
        pred = rf_reg.predict(X_temp)
        pdp_values.append(pred.mean())
    
    ax.plot(feature_range, pdp_values, linewidth=2, color='blue')
    ax.fill_between(feature_range, pdp_values, alpha=0.3, color='blue')
    ax.set_xlabel(f'Feature {feature_idx}', fontsize=12)
    ax.set_ylabel('Partial Dependence', fontsize=12)
    ax.set_title(f'PDP - Feature {feature_idx}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Partial Dependence Plots (상위 3개 피처)\n(부분 의존성 플롯)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('xai_results/06_pdp.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: xai_results/06_pdp.png")

# ============================================================================
# XAI 7: ICE (Individual Conditional Expectation)
# ============================================================================
print("\n[XAI 7] ICE Plot 생성...")
fig, ax = plt.subplots(figsize=(12, 8))
feature_idx = top_3_indices[0]
feature_range = np.linspace(X.iloc[:, feature_idx].min(), 
                           X.iloc[:, feature_idx].max(), 30)

# 샘플 20개에 대한 ICE
n_ice_samples = 20
ice_samples = np.random.choice(len(X_shap_input), n_ice_samples, replace=False)

for sample_idx in ice_samples:
    ice_values = []
    for val in feature_range:
        X_temp = X_shap_input[sample_idx:sample_idx+1].copy()
        X_temp[0, feature_idx] = val
        pred = rf_reg.predict(X_temp)
        ice_values.append(pred[0])
    ax.plot(feature_range, ice_values, alpha=0.3, linewidth=1, color='gray')

# 평균 ICE (PDP)
pdp_values = []
for val in feature_range:
    X_temp = X_shap_input.copy()
    X_temp[:, feature_idx] = val
    pred = rf_reg.predict(X_temp)
    pdp_values.append(pred.mean())

ax.plot(feature_range, pdp_values, linewidth=3, color='red', label='PDP (Average)')
ax.set_xlabel(f'Feature {feature_idx}', fontsize=12)
ax.set_ylabel('Predicted Value', fontsize=12)
ax.set_title(f'ICE Plot - Feature {feature_idx}\n(개별 조건부 기대값 플롯)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xai_results/07_ice.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: xai_results/07_ice.png")

# ============================================================================
# XAI 8: Feature Interaction
# ============================================================================
print("\n[XAI 8] Feature Interaction Plot 생성...")
# 상위 2개 피처 간 상호작용
top_2_indices = top_3_indices[:2]
feature_1 = top_2_indices[0]
feature_2 = top_2_indices[1]

# 2D PDP
n_grid = 20
range_1 = np.linspace(X.iloc[:, feature_1].min(), X.iloc[:, feature_1].max(), n_grid)
range_2 = np.linspace(X.iloc[:, feature_2].min(), X.iloc[:, feature_2].max(), n_grid)

interaction_matrix = np.zeros((n_grid, n_grid))
for i, val_1 in enumerate(range_1):
    for j, val_2 in enumerate(range_2):
        X_temp = X_shap_input.copy()
        X_temp[:, feature_1] = val_1
        X_temp[:, feature_2] = val_2
        pred = rf_reg.predict(X_temp)
        interaction_matrix[i, j] = pred.mean()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(interaction_matrix, cmap='viridis', aspect='auto', origin='lower')
ax.set_xticks(np.arange(0, n_grid, 5))
ax.set_xticklabels([f'{val:.2f}' for val in range_2[::5]], rotation=45)
ax.set_yticks(np.arange(0, n_grid, 5))
ax.set_yticklabels([f'{val:.2f}' for val in range_1[::5]])
ax.set_xlabel(f'Feature {feature_2}', fontsize=12)
ax.set_ylabel(f'Feature {feature_1}', fontsize=12)
ax.set_title(f'Feature Interaction Plot\n(피처 상호작용 플롯)', 
             fontsize=16, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Predicted Value')
plt.tight_layout()
plt.savefig('xai_results/08_feature_interaction.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장 완료: xai_results/08_feature_interaction.png")

# ============================================================================
# 결과 요약 및 해석 문서 생성
# ============================================================================
print("\n[4단계] 결과 해석 문서 생성...")

interpretation = f"""
# 종합 시각화 및 XAI 분석 결과 해석

## 데이터 개요
- 샘플 수: {len(df)}
- 피처 수: {len(feature_cols)}
- 좌표 범위: 위도 [{coords.iloc[:, 0].min():.2f}, {coords.iloc[:, 0].max():.2f}], 
          경도 [{coords.iloc[:, 1].min():.2f}, {coords.iloc[:, 1].max():.2f}]

## 시각화 결과 해석

### 1. 상관관계 Heatmap
- 피처 간 상관관계를 색상으로 표현
- 강한 양의 상관관계(빨간색)와 강한 음의 상관관계(파란색) 확인
- 유사한 피처 그룹 식별 가능

### 2. PCA 시각화
- 설명 분산 비율:
  * PC1: {pca.explained_variance_ratio_[0]:.2%}
  * PC2: {pca.explained_variance_ratio_[1]:.2%}
  * PC3: {pca.explained_variance_ratio_[2]:.2%}
- 총 설명 분산: {pca.explained_variance_ratio_[:3].sum():.2%}
- 3D 공간에서 클러스터 패턴 확인 가능

### 3. 피처 그룹화 시각화
- 5개 그룹으로 분류: Spectral, Temporal, Rhythmic, Harmonic, Timbral
- 각 그룹별 클러스터 간 차이 확인
- 그룹별 특성 파악 가능

### 4. 모델 기반 중요도
- 상위 10개 중요 피처:
{chr(10).join([f'  {i+1}. Feature {f} (Importance: {imp:.4f})' for i, (f, imp) in enumerate(zip(top_10['feature'], top_10['importance']))])}
- 중요도 기반 필터링으로 효율적 분석 가능

### 5. GeoPandas Choropleth
- 지리적 분포 패턴 확인
- 그리드 기반 집계로 지역별 특성 파악
- 분위수 범례로 분포 이해 용이

### 6. 인터랙티브 대시보드
- 모든 시각화를 하나의 대시보드로 통합
- 사용자 상호작용 가능
- 다양한 각도에서 데이터 탐색 가능

## XAI 분석 결과 해석

### 1. SHAP Summary Plot
- 각 피처의 평균 절대 SHAP 값으로 중요도 측정
- 양수/음수 영향 모두 시각화
- 피처별 예측 기여도 정량화

### 2. SHAP Waterfall Plot
- 개별 샘플에 대한 예측 과정 시시각화
- 각 피처가 예측에 미치는 영향 순서대로 표시
- 예측 근거 설명

### 3. SHAP Force Plot
- 개별 샘플의 피처별 영향력 시각화
- 양수/음수 기여도 색상으로 구분
- 여러 샘플 비교 가능

### 4. SHAP Dependence Plot
- 특정 피처의 영향력과 다른 피처와의 상호작용 확인
- 비선형 관계 파악
- 피처 간 의존성 발견

### 5. LIME
- 지역적 설명 (Local Explanation)
- 개별 샘플에 대한 해석 가능
- 피처별 기여도 설명

### 6. Partial Dependence Plot (PDP)
- 피처 값 변화에 따른 예측 평균 변화
- 전역적 패턴 이해
- 비선형 관계 확인

### 7. ICE Plot
- 개별 샘플의 조건부 기대값
- 샘플 간 변이성 확인
- PDP와 함께 사용하여 전체 및 개별 패턴 파악

### 8. Feature Interaction
- 두 피처 간 상호작용 효과 시각화
- 복잡한 관계 파악
- 2D 공간에서의 예측 패턴 확인

## 주요 인사이트

1. **차원 축소**: PCA로 70개 피처를 3차원으로 축소하여 {pca.explained_variance_ratio_[:3].sum():.2%}의 분산 설명

2. **중요 피처**: 상위 10개 피처가 전체 예측의 상당 부분 기여

3. **지리적 패턴**: 좌표 기반 시각화로 지역별 특성 확인 가능

4. **설명 가능성**: XAI 기법으로 모델의 예측 근거 명확히 설명

5. **효율성**: 중요도 기반 필터링으로 분석 효율 향상

## 결론

다양한 시각화 기법과 XAI 방법을 통해 데이터의 구조와 패턴을 효과적으로 이해할 수 있었습니다.
특히 인터랙티브 대시보드와 GeoPandas 시각화를 통해 사용자 친화적인 분석 환경을 구축했습니다.
"""

with open('analysis_interpretation.md', 'w', encoding='utf-8') as f:
    f.write(interpretation)

print("=" * 80)
print("모든 분석 완료!")
print("=" * 80)
print("\n생성된 파일:")
print("  시각화:")
print("    - visualizations/01_correlation_heatmap.png")
print("    - visualizations/02_pca_2d.png")
print("    - visualizations/02_pca_3d.html")
print("    - visualizations/03_feature_groups.png")
print("    - visualizations/04_feature_importance.png")
print("    - visualizations/04_important_features_heatmap.png")
print("    - visualizations/05_geopandas_choropleth.png")
if HAS_MAPCLASSIFY:
    print("    - visualizations/05_geopandas_choropleth_quantiles.png")
else:
    print("    - visualizations/05_geopandas_choropleth_quantiles.png (생략됨: 'mapclassify' 미설치)")
if world_map_created:
    print("    - visualizations/05_world_map.png")
else:
    print("    - visualizations/05_world_map.png (생성 실패)")
print("    - visualizations/06_interactive_dashboard.html")
print("\n  XAI 결과:")
print("    - xai_results/01_shap_summary_bar.png")
print("    - xai_results/01_shap_summary_dot.png")
print("    - xai_results/02_shap_waterfall.png")
print("    - xai_results/03_shap_force.png")
print("    - xai_results/03_shap_force.html")
print("    - xai_results/04_shap_dependence.png")
print("    - xai_results/05_lime_explanation.png")
print("    - xai_results/05_lime_explanation.html")
print("    - xai_results/06_pdp.png")
print("    - xai_results/07_ice.png")
print("    - xai_results/08_feature_interaction.png")
print("\n  문서:")
print("    - analysis_interpretation.md")
print("\n" + "=" * 80)

