import pandas as pd
import numpy as np

# 시드 고정으로 재현성 확보
np.random.seed(42)

# 100개월치 월별 날짜 생성
months = pd.date_range(start='2015-01-01', periods=100, freq='M')

# 더미 데이터 생성
data = {
    '원달러환율': np.random.uniform(1000, 1300, 100),
    'GDP_성장률': np.random.uniform(1.5, 4.5, 100),
    '경기선행지수': np.random.uniform(95, 105, 100),
    '금_가격': np.random.uniform(1200, 2000, 100),
    '천연가스_가격': np.random.uniform(2, 6, 100),
    '원유_가격': np.random.uniform(40, 80, 100),
    '감성지수': np.random.uniform(90, 110, 100),
    '수출량': np.random.uniform(40000, 60000, 100)
}

df = pd.DataFrame(data, index=months)

# CSV로 저장
df.to_csv('your_data.csv', index=False)
print("더미 데이터 'your_data.csv' 파일 저장 완료.")