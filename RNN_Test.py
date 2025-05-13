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
    
    return X, y, scaler

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

# 메인 실행 코드
def main():
    # 데이터 로드 및 전처리
    sequence_length = 12  # 12개월의 데이터로 다음 달 예측
    X, y, scaler = load_and_preprocess_data('your_data.csv', sequence_length)
    
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

if __name__ == "__main__":
    main() 