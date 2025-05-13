# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 숫자 포맷팅 함수
def format_number(x):
    return format(int(x), ',')

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path, sequence_length=12):
    # CSV 파일 로드
    df = pd.read_csv(file_path)
    
    # 날짜 형식 변환
    df['date'] = pd.to_datetime(df['date'])
    
    # 특성과 타겟 분리 (export_restored를 타겟으로 사용)
    features = ['date', 'gdp_growth', 'korea_lead', 'brent_price', 'exchange_rate',
                'gold_price', 'gas_price', 'usa_lead', 'china_lead', 'export', 'export_restored']
    
    # 데이터프레임에서 특성 추출
    dates = df['date'].values
    data = df[features[1:]].values  # date 컬럼 제외한 수치형 데이터만 추출
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 시퀀스 데이터 생성 (타겟으로 export_restored 사용)
    X, y, sequence_dates = [], [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, -1])
        sequence_dates.append(pd.Timestamp(dates[i + sequence_length]))
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler, df[features], sequence_dates

# RNN 모델 생성 함수
def create_rnn_model(sequence_length, n_features):
    model = tf.keras.Sequential([
        # 첫 번째 LSTM 층
        tf.keras.layers.LSTM(256, activation='tanh', input_shape=(sequence_length, n_features), 
                           return_sequences=True, recurrent_dropout=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # 두 번째 LSTM 층
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True, recurrent_dropout=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # 세 번째 LSTM 층
        tf.keras.layers.LSTM(64, activation='tanh', recurrent_dropout=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Dense 층들
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model

def main():
    # 데이터 로드 및 전처리
    sequence_length = 24  # 24개월의 데이터로 다음 달 예측
    X, y, scaler, features_df, sequence_dates = load_and_preprocess_data('data_integ.csv', sequence_length)
    
    # 학습(80%)/테스트(20%) 세트 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = sequence_dates[:split_idx], sequence_dates[split_idx:]
    
    # 모델 생성 및 학습
    model = create_rnn_model(sequence_length, X.shape[2])
    
    # Early Stopping 설정
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=30,
        restore_best_weights=True
    )
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 테스트 세트에 대한 예측
    test_predictions = model.predict(X_test)
    
    # 예측값을 원래 스케일로 변환
    test_predictions_scaled = np.zeros((len(test_predictions), scaler.n_features_in_))
    test_predictions_scaled[:, -1] = test_predictions.flatten()
    test_predictions_original = scaler.inverse_transform(test_predictions_scaled)[:, -1]
    
    # 실제값을 원래 스케일로 변환
    test_actual_scaled = np.zeros((len(y_test), scaler.n_features_in_))
    test_actual_scaled[:, -1] = y_test
    test_actual_original = scaler.inverse_transform(test_actual_scaled)[:, -1]
    
    # MAE 계산
    mae = np.mean(np.abs(test_predictions_original - test_actual_original))
    print(f'\n테스트 세트 MAE: {format_number(mae)}')
    
    # 결과 시각화
    plt.figure(figsize=(15, 10))
    
    # 학습 손실 그래프
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='학습 손실')
    plt.title('모델 학습 손실')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    
    # 테스트 세트 예측 vs 실제 그래프
    plt.subplot(2, 1, 2)
    
    # 날짜 레이블 생성
    test_dates = [d.strftime('%Y-%m') for d in dates_test]
    
    # 실제 데이터와 예측 데이터 플롯
    plt.plot(test_dates, test_actual_original, label='실제 수출량', marker='o')
    plt.plot(test_dates, test_predictions_original, label='예측 수출량', marker='o', linestyle='--')
    
    # y축 값 포맷팅
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.title('테스트 세트: 실제 수출량 vs 예측 수출량')
    plt.xlabel('년-월')
    plt.ylabel('수출량')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 일부 데이터 포인트에 대해 값 표시
    for i in range(0, len(test_dates), 3):  # 3개월 간격으로 값 표시
        plt.text(i, test_actual_original[i], f'\n{format_number(int(test_actual_original[i]))}', 
                ha='center', va='top')
        plt.text(i, test_predictions_original[i], f'{format_number(int(test_predictions_original[i]))}\n', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 마지막 24개의 테스트 데이터에 대한 상세 비교 출력
    print("\n마지막 24개월 실제 수출량 vs 예측 수출량 비교:")
    last_24_idx = -24
    for date, actual, pred in zip(test_dates[last_24_idx:], 
                                test_actual_original[last_24_idx:], 
                                test_predictions_original[last_24_idx:]):
        print(f"{date}: 실제 {format_number(int(actual))} vs 예측 {format_number(int(pred))} "
              f"(오차: {format_number(int(abs(actual - pred)))})")

if __name__ == "__main__":
    main() 