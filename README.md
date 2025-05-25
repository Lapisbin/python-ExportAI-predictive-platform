# ExportAI: 수출 대응형 예측 플랫폼 📦📊

**Repository Name:** `python-ExportAI-predictive-platform`  
**Tech Stack:** Python, XGBoost, LSTM, Pandas, Scikit-learn, BERT, BeautifulSoup, Open
<br>**📘 Notion:** [프로젝트 노션 페이지 바로가기](https://www.notion.so/DSBA-1e8671bafae780f68b3fedea2bf10f45)

## 📌 프로젝트 개요

> 글로벌 경제지표, 정책, 뉴스 등 다양한 데이터를 분석해 **부산항 수출량을 예측**하고,  
> 정부 및 기업의 **의사 결정**을 지원하는 AI 기반 **수출 대응형 예측 플랫폼** 개발 프로젝트입니다.

## 🎯 프로젝트 목표

- 뉴스, 경제지표, 정책, 정세 등 다양한 정형·비정형 데이터를 기반으로 수출량 예측
- XGBoost와 LSTM 모델을 활용한 시계열 기반 예측 정확도 향상
- 중소/대기업의 전략적 수출입 대응을 위한 정량적 정보 제공

## 🛠 기술 스택

| 분야           | 사용 기술                                       |
|----------------|------------------------------------------------|
| 프로그래밍 언어 | Python                                          |
| 모델링         | XGBoost, LSTM, Scikit-learn                     |
| 전처리         | Pandas, Numpy, BeautifulSoup, re               |
| 감성 분석      | BERT (KoBERT), Transformers                     |
| 시각화         | Matplotlib, Seaborn, Plotly (예정)             |

## 📁 주요 기능

- ✅ 경제/정책/금리/정세 등 정형 데이터 통합
- ✅ 뉴스 크롤링 및 감성 분석 점수화
- ✅ 종합 데이터셋 기반 수출량 예측 모델 구축 (XGBoost / LSTM)
- ✅ 예측 결과 시각화 및 리포트 제공 (예정)

## 🗂 데이터셋 정보

| 구분              | 설명                                               |
|-------------------|----------------------------------------------------|
| 타겟 변수         | 부산항 수출 물동량 (2010.01 ~ 2025.03)            |
| 주요 변수         | 원달러 환율, 국내 GDP, 원자재 가격(금, 천연가스, 원유), 뉴스 감성점수 등 |

> 📦 일부 데이터 출처: 한국은행 ECOS, 부산항만공사, Investing.com 등

## 📌 프로젝트 진행 상태

- [✅] 데이터 수집 및 통합
- [✅] 뉴스 크롤링 및 전처리
- [✅] 감성 분석 및 뉴스 점수화
- [✅] XGBoost, LightGBM, LSTM, SARIMAX, TFT 모델 개발 및 학습습
- [✅] 시각화 및 결과 분석

## 👥 팀원 역할 분담

| GitHub ID | 역할 |
|-----------|------|
| [AOTWNetZ](https://github.com/AOTWNetZ) | 뉴스 웹크롤링 및 라벨링 |
| [Kimhyun13in](https://github.com/Kimhyun13in) | 뉴스 키워드 크롤링, TFT 모델 구성|
| [jade-kang](https://github.com/jade-kang) | 데이터셋 통합 및 전처리, EDA |
| [Lapisbin](https://github.com/Lapisbin) | XGBoost, LightGBM, LSTM, SARIMAX 모델 구성 |


## 📄 License

이 프로젝트는 팀 프로젝트로, 외부 상업적 용도로 사용할 수 없습니다.

---

