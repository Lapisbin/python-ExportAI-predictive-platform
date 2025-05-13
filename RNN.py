import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path, sequence_length=12):
    # CSV 파일 로드
    df = pd.read_csv(file_path)
    
    # 특성과 타겟 분리
    features = ['원달러환율', 'GDP_성장률', '경기선행지수', '금_가격', 
                '천연가스_가격', '원유_가격', '감성지수', '수출량']
    data = df[features].values
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 시퀀스 데이터 생성
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, -1])  # 수출량만 타겟으로 사용
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler, df[features]

# RNN 모델 생성 함수
def create_rnn_model(sequence_length, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length, n_features), 
                           return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 미래 수출량 예측 함수
def predict_future_exports(model, last_sequence, scaler, n_future=6):
    """
    미래 n_future 개월의 수출량을 예측합니다.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        # 현재 시퀀스로 다음 달 예측
        current_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        
        # 예측값을 원래 스케일로 변환
        last_row = current_sequence[-1].copy()
        last_row[-1] = current_pred[0][0]  # 수출량 위치의 값을 예측값으로 교체
        inverse_scaled = scaler.inverse_transform(last_row.reshape(1, -1))
        future_predictions.append(inverse_scaled[0][-1])
        
        # 시퀀스 업데이트
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = last_row
    
    return future_predictions

def main():
    # 데이터 로드 및 전처리
    sequence_length = 12  # 12개월의 데이터로 다음 달 예측
    X, y, scaler, features_df = load_and_preprocess_data('your_data.csv', sequence_length)
    
    # 학습/검증/테스트 세트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 모델 생성 및 학습
    model = create_rnn_model(sequence_length, X.shape[2])
    
    # Early Stopping 설정
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 모델 평가
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n테스트 MAE: {test_mae:.4f}')
    
    # 미래 수출량 예측
    last_sequence = X[-1]  # 마지막 시퀀스 데이터
    future_predictions = predict_future_exports(model, last_sequence, scaler)
    
    print("\n향후 6개월 수출량 예측:")
    for i, pred in enumerate(future_predictions, 1):
        print(f"{i}개월 후 예측 수출량: {pred:.2f}")
    
    # 결과 시각화
    plt.figure(figsize=(15, 6))
    
    # 학습 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('모델 학습 손실')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 실제 vs 예측 그래프
    plt.subplot(1, 2, 2)
    # 마지막 12개월의 실제 수출량
    actual_exports = scaler.inverse_transform(X[-12:, -1, -1].reshape(-1, 1))
    months = range(len(actual_exports))
    plt.plot(months, actual_exports, label='실제 수출량', marker='o')
    
    # 예측된 미래 수출량
    future_months = range(len(actual_exports)-1, len(actual_exports) + len(future_predictions)-1)
    plt.plot(future_months, future_predictions, label='예측 수출량', marker='o', linestyle='--')
    
    plt.title('수출량 예측 결과')
    plt.xlabel('월')
    plt.ylabel('수출량')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 