import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 불러오기
df = pd.read_csv('scaled_df.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# 피처 설정
lag_periods = [1, 2, 3, 6, 12]
for lag in lag_periods:
    df[f'lag_{lag}'] = df['export_restored'].shift(lag)

df = df.dropna()

# feature_columns에서 target은 제외
feature_columns = [f'lag_{lag}' for lag in lag_periods]
target_column = 'export_restored'

# feature 정규화 (target은 제외)
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# 데이터 분할
test_start = '2025-01-01'
train = df[df.index < test_start]
test = df[df.index >= test_start]

X_train = train[feature_columns].values
y_train = train[target_column].values

X_test = test[feature_columns].values
y_test = test[target_column].values

# LSTM 입력 형태 맞추기 (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], len(lag_periods), 1))
X_test = X_test.reshape((X_test.shape[0], len(lag_periods), 1))

# 모델 구성
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(len(lag_periods), 1)))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 학습
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# 예측
y_pred = model.predict(X_test).flatten()

# 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"✅ MAE: {mae:.2f}")
print(f"✅ MSE: {mse:.2f}")
print(f"✅ MAPE: {mape*100:.2f}%")

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(test.index, y_test, label='실제 수출량', marker='o')
plt.plot(test.index, y_pred, label='예측 수출량', marker='x')
plt.legend()
plt.title("수출량 예측 (LSTM)")
plt.xlabel("날짜")
plt.ylabel("수출량")
plt.grid(True)
plt.tight_layout()
plt.show()
