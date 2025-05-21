import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.patheffects as pe
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('feat_imp_news_df.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Lag 피처 생성 함수
def create_lag_features(df, columns, lags):    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

# Lag 피처 생성
lag_columns = ['export_restored']
lag_periods = [1, 2, 3, 6, 12]
all_data = create_lag_features(df.copy(), lag_columns, lag_periods)

# 특성 컬럼 정의
# 전체 컬럼에서 제외할 컬럼을 정의
exclude_columns = ['export', 'export_restored']

# 제외한 나머지 컬럼을 base_features로 설정
base_features = [col for col in df.columns if col not in exclude_columns]
lag_features = [f'export_restored_lag{lag}' for lag in lag_periods]
feature_columns = base_features + lag_features

# 학습/테스트 데이터 분리
train_data = all_data.loc[:'2024-09'].copy().dropna()
test_data = all_data.loc['2024-10':'2025-03'].copy()

X_train = train_data[feature_columns]
y_train = train_data['export_restored']
X_test = test_data[feature_columns]
y_test = test_data['export_restored']

# LightGBM 모델 학습
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    random_state=42
)

print("\n=== 모델 학습 시작 ===")
model.fit(X_train, y_train)
print("모델 학습 완료")

# 예측 수행
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy_like = 100 - mape  # 백분율 기준의 정확도 느낌

# 추세 유사성
pearson_corr, _ = pearsonr(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)

print("\n=== 모델 성능 평가 ===")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"예측 정확도(유사): {accuracy_like:.2f}%") # 모델 간 비교용 예측 정확도
print(f"Pearson 상관계수: {pearson_corr:.4f}")
print(f"Spearman 상관계수: {spearman_corr:.4f}")
# 예측 결과 저장 및 오차율 계산
results_df = pd.DataFrame({
    'date': test_data.index,
    '실제 수출액': y_test,
    '예측 수출액': y_pred,
})
results_df['오차율(%)'] = ((results_df['실제 수출액'] - results_df['예측 수출액']) / results_df['실제 수출액']) * 100

# 출력
print("\n=== 2024년 10월-2025년 3월 수출액 예측 결과 ===")
pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
print("\n[단위: USD]")
print("=" * 80)
for idx, row in results_df.iterrows():
    print(f"날짜: {idx.strftime('%Y년 %m월')}")
    print(f"실제 수출액: {row['실제 수출액']:>15,.0f}")
    print(f"예측 수출액: {row['예측 수출액']:>15,.0f}")
    print(f"오차율: {row['오차율(%)']:>18.2f}%")
    print("-" * 80)

print(f"평균 오차율: {results_df['오차율(%)'].mean():>16.2f}%")
print("=" * 80)

# 그래프 시각화
plt.figure(figsize=(9, 8))
ax = plt.gca()

# 실제값 라인 (그림자 효과)
ax.plot(
    results_df['date'], results_df['실제 수출액'],
    label='실제값',
    marker='o', linestyle='-', linewidth=4, markersize=8,
    color='#003366', markerfacecolor='#003366', markeredgecolor='white',
    path_effects=[pe.Stroke(linewidth=6, foreground='white', alpha=0.3), pe.Normal()]
)
# 예측값 라인 (그림자 효과)
ax.plot(
    results_df['date'], results_df['예측 수출액'],
    label='예측값',
    marker='s', linestyle='--', linewidth=4, markersize=12,
    color='#FF9900', markerfacecolor='#FF9900', markeredgecolor='#003366', markeredgewidth=2, alpha=0.95,
    path_effects=[pe.Stroke(linewidth=6, foreground='white', alpha=0.3), pe.Normal()]
)

# 값 표시 (그림자 효과)
for idx, row in results_df.iterrows():
    ax.annotate(
        f'{row["실제 수출액"]:,.0f}', xy=(row['date'], row["실제 수출액"]),
        xytext=(0, 12), textcoords='offset points',
        ha='center', va='bottom', fontsize=19, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2, alpha=1.0),
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )
    ax.annotate(
        f'{row["예측 수출액"]:,.0f}', xy=(row['date'], row["예측 수출액"]),
        xytext=(0, -18), textcoords='offset points',
        ha='center', va='top', fontsize=19, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2, alpha=1.0),
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )

# 타이틀/축/범례
plt.title(
    '2024년 10월 ~ 2025년 3월\n수출량 예측 결과 (정량+정성 분석)',
    fontsize=20, fontweight='bold', y=0.9
)
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=14)
ax.tick_params(axis='x', pad=8)
ax.tick_params(axis='y', pad=8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('#CCCCCC')
plt.gca().spines['bottom'].set_color('#CCCCCC')

# y축만 점선 그리드
plt.grid(axis='y', linestyle='--', color='#CCCCCC', alpha=0.6)
plt.grid(axis='x', visible=False)

# 범례 스타일
plt.legend(
    fontsize=14, loc='upper left',
    frameon=True, facecolor='white', edgecolor='#CCCCCC',
    fancybox=True, framealpha=1.0
)

plt.margins(y=0.8)
plt.tight_layout()
plt.savefig('export_plot.png', transparent=True, dpi=300)
plt.show()

# 2. 오차율
plt.subplot(2, 1, 2)
error_percentage = ((results_df['실제 수출액'] - results_df['예측 수출액']) / results_df['실제 수출액']) * 100
plt.plot(results_df['date'], error_percentage, color='red', marker='o')
plt.title('예측 오차율 (%)')
plt.xlabel('날짜')
plt.ylabel('오차율 (%)')
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()

# 이동평균으로 추세선 시각화
results_df['실제_이동평균'] = results_df['실제 수출액'].rolling(window=2).mean()
results_df['예측_이동평균'] = results_df['예측 수출액'].rolling(window=2).mean()

plt.figure(figsize=(10, 6))
plt.plot(results_df.index, results_df['실제_이동평균'], label='실제 추세선', linestyle='--')
plt.plot(results_df.index, results_df['예측_이동평균'], label='예측 추세선', linestyle='--')
plt.legend()
plt.title('예측 vs 실제 수출액 추세선 (이동평균 기반)')
plt.grid(True)
plt.xticks(rotation=45)

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({
    '특성': feature_columns,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=True)
plt.barh(importance_df['특성'], importance_df['중요도'])
plt.title('특성 중요도')
plt.xlabel('중요도')
plt.show()
