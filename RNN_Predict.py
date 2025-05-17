import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 및 전처리
df = pd.read_csv('scaled_df.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# lag feature 생성 함수
def create_lag_features(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

lag_columns = ['export_restored']
lag_periods = [1, 2, 3, 6, 12]
all_data = create_lag_features(df.copy(), lag_columns, lag_periods)

base_features = ['gdp_growth', 'exchange_rate', 'gold_price', 'gas_price',
                 'korea_lead', 'usa_lead', 'china_lead', 'brent_price', 'trade']
lag_features = [f'export_restored_lag{lag}' for lag in lag_periods]
feature_columns = base_features + lag_features

# train / test 분리
train_data = all_data.loc[:'2024-12'].dropna().copy()
test_data = all_data.loc['2025-01':'2025-03'].copy()

# 시퀀스 생성 함수 (RNN 입력 형태)
def create_sequences(data, feature_cols, target_col, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[feature_cols].iloc[i:i+time_steps].values)
        y.append(data[target_col].iloc[i+time_steps])
    return np.array(X), np.array(y)

# train 데이터 시퀀스 생성
X_train, y_train = create_sequences(train_data, feature_columns, 'export_restored', time_steps=3)

# test 데이터 시퀀스 생성 (test는 dropna 아니므로 lag 때문에 데이터가 부족할 수 있음)
# test 데이터를 train+test 합친 데이터로 시퀀스 생성하는 게 안정적
combined_data = all_data.loc[:'2025-03'].copy()
combined_data = combined_data.dropna()
X_combined, y_combined = create_sequences(combined_data, feature_columns, 'export_restored', time_steps=3)

# test set이 combined 내에 포함되므로 test 인덱스 위치 찾기
test_start_idx = combined_data.index.get_loc(pd.to_datetime('2025-01-01'))

# test 데이터 시퀀스는 combined 데이터 중 test 시작 이후만 추출
test_start_seq_idx = test_start_idx - 3  # time_steps만큼 앞당김 (시퀀스 생성시 앞부분 제외됨)
X_test = X_combined[test_start_seq_idx:]
y_test = y_combined[test_start_seq_idx:]

# 모델 정의
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 학습 조기 종료 콜백
early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

print("\n=== 모델 학습 시작 ===")
history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, callbacks=[early_stop])
print("모델 학습 완료")

# 예측
y_pred = model.predict(X_test).flatten()

# 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy_like = 100 - mape

print("\n=== 모델 성능 평가 ===")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"예측 정확도(유사): {accuracy_like:.2f}%")

# 결과 데이터프레임 생성
results_df = pd.DataFrame({
    'date': combined_data.index[test_start_seq_idx + 3:],  # 시퀀스 offset 보정
    '실제 수출액': y_test,
    '예측 수출액': y_pred,
})
results_df.set_index('date', inplace=True)

# 오차율 계산
results_df['오차율(%)'] = ((results_df['실제 수출액'] - results_df['예측 수출액']) / results_df['실제 수출액']) * 100

# 예측 결과 출력
print("\n=== 2025년 1-3월 수출액 예측 결과 ===")
pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else str(x))
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

# 시각화
plt.figure(figsize=(15, 10))

# 1. 실제값과 예측값 비교 그래프
plt.subplot(2, 1, 1)
plt.plot(results_df.index, results_df['실제 수출액'], label='실제값', marker='o')
plt.plot(results_df.index, results_df['예측 수출액'], label='예측값', marker='s')

for idx, row in results_df.iterrows():
    plt.annotate(f'{row["실제 수출액"]:,.0f}', xy=(row.name, row["실제 수출액"]),
                 xytext=(0, 10), textcoords='offset points', ha='center', va='bottom')
    plt.annotate(f'{row["예측 수출액"]:,.0f}', xy=(row.name, row["예측 수출액"]),
                 xytext=(0, -15), textcoords='offset points', ha='center', va='top')

plt.title('2025년 1-3월 수출액 예측 결과')
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
plt.plot(results_df.index, results_df['오차율(%)'], color='red', marker='o')
plt.title('예측 오차율 (%)')
plt.xlabel('날짜')
plt.ylabel('오차율 (%)')
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 학습 손실 및 MAE 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('손실 (Loss)')
plt.xlabel('에포크')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.title('MAE')
plt.xlabel('에포크')
plt.ylabel('Mean Absolute Error')
plt.grid(True)
plt.tight_layout()
plt.show()
