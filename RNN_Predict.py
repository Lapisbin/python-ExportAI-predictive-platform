import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import pearsonr, spearmanr

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('feat_imp_news_df.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# lag feature 생성 함수
def create_lag_features(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

lag_columns = ['export']
lag_periods = [1, 2, 3, 6, 12]
all_data = create_lag_features(df.copy(), lag_columns, lag_periods)

# 특성 컬럼 정의
# 전체 컬럼에서 제외할 컬럼을 정의
exclude_columns = ['export', 'export_restored']

# 제외한 나머지 컬럼을 base_features로 설정
base_features = [col for col in df.columns if col not in exclude_columns]

lag_features = [f'export_lag{lag}' for lag in lag_periods]
feature_columns = base_features + lag_features

# 데이터 분리
train_data = all_data.loc[:'2024-09'].dropna().copy()
test_data = all_data.loc['2024-10':'2025-03'].copy()
combined_data = pd.concat([train_data, test_data]).dropna()

# 시퀀스 생성 함수
def create_sequences(data, feature_cols, target_col, time_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[feature_cols].iloc[i:i+time_steps].values)
        y.append(data[target_col].iloc[i+time_steps])
    return np.array(X), np.array(y)

# 학습용 시퀀스
X_train, y_train = create_sequences(train_data, feature_columns, 'export', time_steps=6)

# 전체 시퀀스에서 테스트용 시퀀스 분리
X_combined, y_combined = create_sequences(combined_data, feature_columns, 'export', time_steps=6)
test_start_idx = combined_data.index.get_loc(pd.to_datetime('2024-10-01'))
test_start_seq_idx = test_start_idx - 6  # 시퀀스 길이만큼 앞에서 시작
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

# 조기 종료 콜백
early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

print("\n=== 모델 학습 시작 ===")
history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, callbacks=[early_stop])
print("모델 학습 완료")

# 예측
y_pred_scaled = model.predict(X_test).flatten()

# 역정규화를 위해 export와 export_restored 매핑 
# LSTM은 가중치 초기화, 경사하강법 최적화 과정에서 입력값 범위가 매우 중요 (XGBoost, LightGBM, SARIMAX와 다르게 모델 자체가 데이터 전체를 보고 학습하기 때문에 데이터 전체의 범위가 중요)
restore_lookup = combined_data['export_restored'].values[6 + test_start_seq_idx:]  # 시퀀스 offset 보정
scale_lookup = combined_data['export'].values[6 + test_start_seq_idx:]

# 평균, 표준편차 복원용 (표준 정규화 기준일 경우)
mean_export = df['export_restored'].mean()
std_export = df['export_restored'].std()

# 역정규화: 예측값만 역정규화
y_pred = y_pred_scaled * std_export + mean_export
y_true = combined_data['export_restored'].values[6 + test_start_seq_idx:] # 실제값은 export_restored에서 직접 가져오기
# y_true = y_test * std_export + mean_export // 기존 y_true 정확도 98퍼정도 나옴 다만만 수출량이 실제와 좀 다름

# 결과 데이터프레임
results_df = pd.DataFrame({
    'date': combined_data.index[6 + test_start_seq_idx:],
    '실제 수출액': y_true,
    '예측 수출액': y_pred
})
results_df.set_index('date', inplace=True)
results_df['오차율(%)'] = ((results_df['실제 수출액'] - results_df['예측 수출액']) / results_df['실제 수출액']) * 100

# 평가 지표
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
accuracy_like = 100 - mape

# 추세 유사성
pearson_corr, _ = pearsonr(y_true, y_pred)
spearman_corr, _ = spearmanr(y_true, y_pred)


print("\n=== 모델 성능 평가 ===")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"예측 정확도(유사): {accuracy_like:.2f}%")
print(f"📈 Pearson 상관계수: {pearson_corr:.4f}")
print(f"📈 Spearman 상관계수: {spearman_corr:.4f}")

# 예측 출력
print("\n=== 2024년 10월-2025년 3월 수출액 예측 결과 (LSTM) ===")
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
plt.figure(figsize=(16, 8))  # 그래프 크기 좀 더 키움

plt.plot(results_df.index, results_df['실제 수출액'], label='실제값', marker='o', linewidth=2, markersize=10)
plt.plot(results_df.index, results_df['예측 수출액'], label='예측값', marker='s', linewidth=2, markersize=10)

# 값 표시 (annotation) 글씨 크기 키움
for idx, row in results_df.iterrows():
    plt.annotate(f'{row["실제 수출액"]:,.0f}', xy=(idx, row["실제 수출액"]),
                 xytext=(0, 12), textcoords='offset points', ha='center', va='bottom', fontsize=24, fontweight='bold')
    plt.annotate(f'{row["예측 수출액"]:,.0f}', xy=(idx, row["예측 수출액"]),
                 xytext=(0, -18), textcoords='offset points', ha='center', va='top', fontsize=24, fontweight='bold')

plt.title('2024년 10월-2025년 3월 수출액 예측 결과 (정량+정성)', fontsize=24, fontweight='bold')

plt.legend(fontsize=16)
plt.grid(True)
plt.margins(y=0.2)

# x축 눈금 라벨 크기 및 각도 조절
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45, fontsize=30)

# y축 눈금 라벨 크기 조절
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

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
plt.show()

# 학습 손실 시각화
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
