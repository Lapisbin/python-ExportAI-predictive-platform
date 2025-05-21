import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

models = ['XGBoost', 'LightGBM', 'LSTM', 'SARIMAX', 'TFT']
accuracies = [90.42, 94.75, 91.51, 96.66, 50.87]

colors = ['#00A1B8', '#5F82A2', '#33558C', '#5960A6', '#2E328A']

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=colors, width=0.6)

# 바 위에 정확도 값 표시
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}', ha='center', va='bottom', fontsize=15, fontweight='bold')

plt.ylim(0, 100)
plt.ylabel('예측 정확도(%)', fontsize=15, fontweight='bold')
plt.title('모델별 예측 정확도 비교(정량+정성 분석)', fontsize=18, fontweight='bold', pad=15)
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
