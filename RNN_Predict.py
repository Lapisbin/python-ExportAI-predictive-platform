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
        sequence_dates.append(pd.Timestamp(dates[i + sequence_length]))  # numpy.datetime64를 pandas Timestamp로 변환
    
    X = np.array(X)
    y = np.array(y)
    
    # 원본 데이터프레임도 반환
    return X, y, scaler, df[features], sequence_dates

# RNN 모델 생성 함수
def create_rnn_model(sequence_length, n_features):
    model = tf.keras.Sequential([
        # 첫 번째 LSTM 층의 유닛 수를 늘리고 return_sequences를 True로 설정
        tf.keras.layers.LSTM(128, activation='relu', input_shape=(sequence_length, n_features), 
                           return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        
        # 두 번째 LSTM 층 추가
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        
        # 세 번째 LSTM 층
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Dense 층 추가
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # 학습률 조정을 위한 Adam 옵티마이저 설정
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def predict_future_exports(model, last_sequence, last_date, scaler, n_future=6):
    """
    미래 수출량을 예측하는 함수
    
    Args:
        model: 학습된 RNN 모델
        last_sequence: 마지막 시퀀스 데이터
        last_date: 마지막 날짜
        scaler: 데이터 스케일러
        n_future: 예측할 미래 개월 수
    
    Returns:
        미래 수출량 예측값과 예측된 날짜
    """
    future_predictions = []
    prediction_dates = []
    current_sequence = last_sequence.copy()
    current_date = pd.to_datetime(last_date)
    
    for i in range(n_future):
        # 현재 시퀀스로 다음 달 예측
        current_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        future_predictions.append(current_pred[0, 0])
        
        # 다음 달 날짜 계산
        current_date = current_date + pd.DateOffset(months=1)
        prediction_dates.append(current_date)
        
        # 시퀀스 업데이트
        next_sequence = current_sequence.copy()
        next_sequence = np.roll(next_sequence, -1, axis=0)
        next_sequence[-1] = current_sequence[-1]
        next_sequence[-1, -1] = current_pred[0, 0]
        current_sequence = next_sequence
    
    # 예측값을 원래 스케일로 변환
    predictions_scaled = np.zeros((len(future_predictions), scaler.n_features_in_))
    predictions_scaled[:, -1] = future_predictions
    predictions_original = scaler.inverse_transform(predictions_scaled)[:, -1]
    
    return predictions_original, prediction_dates

# 메인 실행 코드
def main():
    # 데이터 로드 및 전처리
    sequence_length = 12  # 12개월의 데이터로 다음 달 예측
    X, y, scaler, features_df, sequence_dates = load_and_preprocess_data('data_integ.csv', sequence_length)
    
    # 학습/검증/테스트 세트 분할
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, sequence_dates, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
        X_train, y_train, dates_train, test_size=0.2, random_state=42)
    
    # 모델 생성 및 학습
    model = create_rnn_model(sequence_length, X.shape[2])
    
    # Early Stopping 설정
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 모델 평가
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n테스트 MAE: {test_mae:.4f}')
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 미래 수출량 예측
    last_sequence = X[-1]
    last_date = sequence_dates[-1]
    future_predictions, prediction_dates = predict_future_exports(model, last_sequence, last_date, scaler, n_future=6)
    
    print("\n향후 6개월 수출량 예측:")
    for date, pred in zip(prediction_dates, future_predictions):
        print(f"{date.strftime('%Y-%m')}: {pred:.2f}")

    # 결과 시각화
    plt.figure(figsize=(15, 6))
    
    # 학습 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='학습 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('모델 학습 손실')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    
    # 실제 vs 예측 그래프
    plt.subplot(1, 2, 2)
    
    # 마지막 12개월의 실제 수출량
    actual_data = np.zeros((12, scaler.n_features_in_))
    actual_data[:, -1] = X[-12:, -1, -1]
    actual_exports = scaler.inverse_transform(actual_data)[:, -1]
    
    # 날짜 레이블 생성
    last_12_dates = pd.to_datetime(sequence_dates[-12:])  # pandas Timestamp로 변환
    future_dates = prediction_dates
    
    # 실제 데이터 플롯
    plt.plot([d.strftime('%Y-%m') for d in last_12_dates], 
             actual_exports, label='실제 수출량', marker='o')
    
    # 예측 데이터 플롯
    plt.plot([d.strftime('%Y-%m') for d in future_dates], 
             future_predictions, label='예측 수출량', marker='o', linestyle='--')
    
    plt.title('수출량 예측 결과')
    plt.xlabel('년-월')
    plt.ylabel('수출량')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 