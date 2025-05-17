import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 데이터 로드
df = pd.read_csv('scaled_df.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 학습/테스트 데이터 분리
test_df = df.loc['2025-01':'2025-03']
train_df = df.loc[:'2024-12']

# Lag 피처 생성 함수
def create_lag_features(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

# Lag 피처 생성
lag_columns = ['export_restored']  # 수출량의 과거 데이터 사용
lag_periods = [1, 2, 3, 6, 12]    # 1,2,3,6,12개월 전의 데이터
all_data = create_lag_features(df.copy(), lag_columns, lag_periods)

# 원래 특성 컬럼 정의
base_features = ['gdp_growth', 'exchange_rate', 'gold_price', 'gas_price', 
                'korea_lead', 'usa_lead', 'china_lead', 'brent_price', 'trade']

# Lag 피처 컬럼 추가
lag_features = [f'export_restored_lag{lag}' for lag in lag_periods]
feature_columns = base_features + lag_features # 경제 지표 컬럼과 수출량의 과거 데이터 컬럼 합치기

# 학습/테스트 데이터 분리
train_data = all_data.loc[:'2024-12'].copy()
test_data = all_data.loc['2025-01':'2025-03'].copy()

# NA 값 처리 (첫 12개월은 lag 피처가 없으므로로)
train_data = train_data.dropna()

X_train = train_data[feature_columns]
y_train = train_data['export_restored']
X_test = test_data[feature_columns]
y_test = test_data['export_restored']

# XGBoost 모델 생성 및 학습
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    random_state=42
)

# 모델 학습
print("\n=== 모델 학습 시작 ===")
model.fit(X_train, y_train)
print("모델 학습 완료")

# 예측 수행
y_pred = model.predict(X_test)

# 모델 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== 모델 성능 평가 ===")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

# 예측 결과를 데이터프레임으로 저장
results_df = pd.DataFrame({
    'date': test_data.index,
    '실제 수출액': y_test,
    '예측 수출액': y_pred,
})

# 오차율 계산
results_df['오차율(%)'] = ((results_df['실제 수출액'] - results_df['예측 수출액']) / results_df['실제 수출액']) * 100

# 예측 결과 출력
print("\n=== 2025년 1-3월 수출액 예측 결과 ===")
pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
print("\n[단위: USD]")
print("=" * 80)
for idx, row in results_df.iterrows():
    print(f"날짜: {idx.strftime('%Y년 %m월')}")
    print(f"실제 수출액: {row['실제 수출액']:>15,.0f}")
    print(f"예측 수출액: {row['예측 수출액']:>15,.0f}")
    print(f"오차율: {row['오차율(%)']:>18.2f}%")
    print("-" * 80)

# 평균 오차율 출력
print(f"평균 오차율: {results_df['오차율(%)'].mean():>16.2f}%")
print("=" * 80)

# 결과 시각화
plt.figure(figsize=(15, 10))

# 1. 실제값과 예측값 비교 그래프
plt.subplot(2, 1, 1)
plt.plot(results_df['date'], results_df['실제 수출액'], label='실제값', marker='o')
plt.plot(results_df['date'], results_df['예측 수출액'], label='예측값', marker='s')

# 값 표시 추가
for idx, row in results_df.iterrows():
    # 실제값 표시 (위쪽)
    plt.annotate(f'{row["실제 수출액"]:,.0f}', 
                xy=(row.name, row["실제 수출액"]),
                xytext=(0, 10),  # 10 포인트 위에 표시
                textcoords='offset points',
                ha='center',
                va='bottom')
    
    # 예측값 표시 (아래쪽)
    plt.annotate(f'{row["예측 수출액"]:,.0f}', 
                xy=(row.name, row["예측 수출액"]),
                xytext=(0, -15),  # 15 포인트 아래에 표시
                textcoords='offset points',
                ha='center',
                va='top')

plt.title('2025년 1-3월 수출액 예측 결과')
plt.xlabel('날짜')
plt.ylabel('수출액')
plt.legend()
plt.grid(True)

# 여백 추가
plt.margins(y=0.2)  # y축 여백을 20% 추가

# x축 날짜 포맷 설정
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# 2. 오차율 그래프
plt.subplot(2, 1, 2)
error_percentage = ((results_df['실제 수출액'] - results_df['예측 수출액']) / results_df['실제 수출액']) * 100
plt.plot(results_df['date'], error_percentage, color='red', marker='o')
plt.title('예측 오차율 (%)')
plt.xlabel('날짜')
plt.ylabel('오차율 (%)')
plt.grid(True)

# x축 날짜 포맷 설정
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# 그래프 레이아웃 조정
plt.tight_layout()

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({
    '특성': feature_columns,
    '중요도': model.feature_importances_
})
importance_df = importance_df.sort_values('중요도', ascending=True)

plt.barh(importance_df['특성'], importance_df['중요도'])
plt.title('특성 중요도')
plt.xlabel('중요도')

# 모든 그래프 표시
plt.show() 