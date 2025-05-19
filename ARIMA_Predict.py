import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 데이터 로드
df = pd.read_csv('feat_imp_df.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# 학습/테스트 분리
train_df = df.loc[:'2024-12'].copy()
test_df = df.loc['2025-01':'2025-03'].copy()

# ARIMA는 타겟 변수만 사용
y_train = train_df['export_restored']
y_test = test_df['export_restored']


# ARIMA 모델 학습 (p=1, d=1, q=1)
model = ARIMA(y_train, order=(1, 1, 1))
results = model.fit()

# 예측
y_pred = results.forecast(steps=len(y_test))

# 평가 지표 계산
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


# 예측 결과를 데이터프레임으로 저장
results_df = pd.DataFrame({
    'date': y_test.index,
    '실제 수출액': y_test.values,
    '예측 수출액': y_pred.values,
})

# 오차율 계산
results_df['오차율(%)'] = ((results_df['실제 수출액'] - results_df['예측 수출액']) / results_df['실제 수출액']) * 100

# 예측 결과 출력
print("\n=== 2025년 1-3월 수출액 예측 결과 (ARIMA) ===")
pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
print("\n[단위: TEU]")
print("=" * 80)
for idx, row in results_df.iterrows():
    print(f"날짜: {row['date'].strftime('%Y년 %m월')}")
    print(f"실제 수출액: {row['실제 수출액']:>15,.0f}")
    print(f"예측 수출액: {row['예측 수출액']:>15,.0f}")
    print(f"오차율: {row['오차율(%)']:>18.2f}%")
    print("-" * 80)

# 평균 오차율 출력
print(f"평균 오차율: {results_df['오차율(%)'].mean():>16.2f}%")
print("=" * 80)

# 결과 시각화
plt.figure(figsize=(15, 10))

# 1. 실제값 vs 예측값
plt.subplot(2, 1, 1)
plt.plot(results_df['date'], results_df['실제 수출액'], label='실제값', marker='o')
plt.plot(results_df['date'], results_df['예측 수출액'], label='예측값', marker='s')

for idx, row in results_df.iterrows():
    plt.annotate(f'{row["실제 수출액"]:,.0f}', 
                 xy=(row['date'], row["실제 수출액"]),
                 xytext=(0, 10), textcoords='offset points', ha='center', va='bottom')
    plt.annotate(f'{row["예측 수출액"]:,.0f}', 
                 xy=(row['date'], row["예측 수출액"]),
                 xytext=(0, -15), textcoords='offset points', ha='center', va='top')

plt.title('2025년 1-3월 수출액 예측 결과 (ARIMA)')
plt.xlabel('날짜')
plt.ylabel('수출액')
plt.legend()
plt.grid(True)
plt.margins(y=0.2)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)


# 2. 오차율 그래프
plt.subplot(2, 1, 2)
plt.plot(results_df['date'], results_df['오차율(%)'], color='red', marker='o')
plt.title('예측 오차율 (%)')
plt.xlabel('날짜')
plt.ylabel('오차율 (%)')
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

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

print(results.summary().tables[1])

plt.tight_layout()
plt.show()
