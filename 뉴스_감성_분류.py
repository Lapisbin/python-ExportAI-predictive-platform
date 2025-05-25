# -*- coding: utf-8 -*-
"""뉴스 감성 분류.ipynb"""

# 필요한 라이브러리 설치
!pip install langchain langchain-community langchain-openai langchain-huggingface langchain-chroma pdfplumber

# Google Drive 마운트 및 경로 확인
from google.colab import drive
drive.mount('/content/drive')
import os
os.getcwd()
os.listdir()

# 작업 디렉토리로 변경
%pwd
%cd /content/drive/MyDrive/항만_프로젝트/corpus

from google.colab import userdata
import os

os.environ["LANGCHAIN_API_KEY"] = userdata.get('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_ENDPOINT"] = userdata.get('LANGSMITH_ENDPOINT')
os.environ["LANGCHAIN_PROJECT"] = userdata.get('LANGSMITH_PROJECT')
os.environ["LANGCHAIN_TRACING"] = userdata.get('LANGSMITH_TRACING')
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# CSV 로더 임포트
from langchain_community.document_loaders import CSVLoader

# 감성분류 학습 데이터 로드
docs = CSVLoader("news_data_25_edu.csv").load()

# 한국어 자연어 처리를 위한 KoNLPy 설치
!python3 -m pip install konlpy

# 한국어 형태소 분석기 초기화
from konlpy.tag import Kkma, Okt
okt = Okt()
kkma = Kkma()

# 한국어 토큰 개수를 세는 함수
def len_okt(text):
  tokens = [token for token in okt.morphs(text)]

  return len(tokens)

# 텍스트 스플리터 설정
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],  # 분할 기준 (문단, 줄바꿈, 공백 순)
    chunk_size=1000,                 # 각 청크의 최대 토큰 수
    chunk_overlap=50,                # 청크 간 겹치는 토큰 수
    length_function=len_okt          # 한국어 토큰 길이 계산 함수
)

# 문서 분할
texts = text_splitter.split_documents(docs)

import getpass
import os

if not userdata.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API Key for OpenAI:")

# OpenAI 임베딩 모델 초기화
from langchain_openai import OpenAIEmbeddings
openai_embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

# 벡터 데이터베이스 저장 디렉토리로 이동
%pwd
%cd /content/drive/MyDrive/항만_프로젝트

# Chroma 벡터 데이터베이스 생성 및 문서 임베딩
from langchain_chroma import Chroma
import os
import shutil
from tqdm import tqdm

save_directory = "./chroma_docs.db"

print('잠시만 기다려주세요.')

# 기존 데이터베이스가 있으면 삭제
if os.path.exists(save_directory):
    shutil.rmtree(save_directory)
    print(f'{save_directory}가 삭제되었습니다.')

print("문서 벡터화 시작합니다.")

# 배치 크기 설정
BATCH_SIZE = 100

# Chroma DB 생성
db = Chroma(embedding_function=openai_embedding_model, persist_directory=save_directory)

# 배치 단위로 처리하여 벡터화 및 저장
total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="배치 처리 중", total=total_batches):
    batch = docs[i:i+BATCH_SIZE]
    # 배치를 DB에 추가 (자동으로 저장됨)
    db.add_documents(documents=batch)
print("새로운 Chroma DB가 생성되었습니다.")

# 유사도 검색을 위한 리트리버 설정
retriever = db.as_retriever(
    search_kwargs={"k": 3},
)

# 감성 분석 프롬프트 정의
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an assistant that categorizes whether the input text is positive, neutral or negative for Korea's exports.
Answer whether the following pieces of retrieved context is positively neutral or negative for Korean exports.
If you don't know the answer, just say that neutral.

Must answer in one word(positive, neutral, negative).

#Context:
{context}
""",

        ),
        ("human", "{question}")
    ]
)

# OpenAI ChatGPT 모델 초기화
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    model = 'gpt-4o-mini',                          # 사용할 모델
    temperature=0,                                  # 결과의 일관성을 위해 0으로 설정
    streaming=True,                                 # 스트리밍 출력 활성화
    callbacks=[StreamingStdOutCallbackHandler()],   # 실시간 출력 콜백
)

# 검색된 문서를를 텍스트로 합치는 함수
def format_docs(docs):
  return "\n\n".join(document.page_content for document in docs)

# RAG 체인 구성
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

chain = {
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
} | prompt | llm | StrOutputParser()

# 감성 분석할 뉴스 데이터 로드
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/news_data_10_24/news_data_10_24.csv')

# tqdm을 사용하여 각 뉴스 기사에 대해 sentiment 분석 수행 및 진행 상황 표시
from tqdm import tqdm

sentiment_results = []

for i in tqdm(range(len(data))):
    question = data.iloc[i]['text'] # 뉴스 본문 추출
    response = chain.invoke(question)   # RAG 체인을 통해 감성 분석 수행
    print('\n')
    sentiment_results.append(response)  # 결과를 리스트에 추가

# 결과를 DataFrame에 새로운 'sentiment' 열로 추가
data['sentiment'] = sentiment_results

# 결과를 CSV 파일로 저장
data.to_csv('/content/drive/MyDrive/news_data_10_24_with_sentiment.csv', index=False)