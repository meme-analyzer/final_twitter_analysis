import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ✅ 경로 설정 (상위 디렉토리에서 config 불러오기 위해 sys.path 추가)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class SeleniumTwitterPreprocessor:
    def __init__(self):
        # ✅ 디렉토리 경로 설정 및 문장 임베딩 모델 로딩
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def load_twitter_data(self, filename):
        # ✅ 원시 트위터 데이터 CSV 로드
        filepath = os.path.join(self.raw_data_dir, filename)
        df = pd.read_csv(filepath)
        print(f"📅 데이터 로드 완료: {len(df)}개 게시물")
        return df

    def preprocess(self, df):
        print("\n🧹=== 트위터 데이터 전처리 시작 ===")

        # ✅ 날짜, 시간, 요일 추출
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.strftime('%Y-%m-%d')  # 문자열로 변환해 저장
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['day_abbr'] = df['created_at'].dt.strftime('%a').str.upper()  # ✅ MON, TUE 형식 요일 추가

        # ✅ 결측값 처리
        df['text'] = df['text'].fillna('')
        df['author'] = df['author'].fillna('[deleted]')

        # ✅ 숫자형 컬럼 처리: 콤마 제거 후 숫자로 변환
        df['likes'] = pd.to_numeric(df['likes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        df['retweets'] = pd.to_numeric(df['retweets'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        df['views'] = pd.to_numeric(df['views'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

        # ✅ 텍스트 정제
        df['text_clean'] = df['text'].apply(self.clean_text)

        # ✅ 참여 점수 계산 (좋아요 + 2*리트윗 + 0.1*조회수)
        df['engagement_score'] = df['likes'] + df['retweets'] * 2 + df['views'] * 0.1

        # ✅ 최신순 정렬
        df = df.sort_values(by='created_at', ascending=False).reset_index(drop=True)

        print(f"✅ 전처리 완료: {len(df)}개 게시물")
        return df

    def clean_text(self, text):
        # ✅ URL 제거 및 공백 정리
        if pd.isna(text) or text == '':
            return ''
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def analyze_temporal_patterns(self, df):
        # ✅ 시간 패턴 분석 (일자별, 시간대별, 요일별)
        print("\n⏱️=== 시간 패턴 분석 ===")
        daily_posts = df.groupby('date').size()
        hourly_dist = df['hour'].value_counts().sort_index()
        day_dist = df['day_of_week'].value_counts().sort_index()
        days = ['월', '화', '수', '목', '금', '토', '일']

        print(f"🗖️ 데이터 기간: {daily_posts.index.min()} ~ {daily_posts.index.max()}")
        print(f"📈 일 평균 게시물 수: {daily_posts.mean():.2f}")
        print(f"⏰ 가장 활발한 시간대: {hourly_dist.idxmax()}시")
        print(f"🗓️ 가장 활발한 요일: {days[day_dist.idxmax()]}요일")

        return {
            'daily_posts': daily_posts,
            'hourly_dist': hourly_dist,
            'day_dist': day_dist
        }

    def perform_clustering(self, df, n_clusters=5):
        # ✅ 문장 임베딩 후 PCA 축소 + KMeans 클러스터링
        print("\n🔗=== 클러스터링 시작 ===")
        embeddings = self.embedder.encode(df['text_clean'].tolist(), show_progress_bar=True)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        df['x'] = reduced[:, 0]
        df['y'] = reduced[:, 1]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(reduced)
        df['cluster'] = kmeans.labels_
        print(f"🎯 클러스터링 완료 (군집 수: {n_clusters})")
        return df

    def estimate_last_seen(self, df):
        # ✅ 동일한 텍스트 기준 마지막 등장 시점 추정
        print("\n🔍=== 생존 분석용 마지막 등장 시점 추정 ===")
        grouped = df.groupby('text_clean')['created_at'].max().reset_index()
        grouped.columns = ['text_clean', 'last_seen_at']
        df = df.merge(grouped, on='text_clean', how='left')
        print("✅ last_seen_at 컬럼 생성 완료")
        return df

    def save_processed_data(self, df, output_filename):
        # ✅ 전처리된 데이터 및 요약 통계 저장
        os.makedirs(self.processed_data_dir, exist_ok=True)
        output_path = os.path.join(self.processed_data_dir, output_filename)
        df.to_csv(output_path, index=False)
        print(f"📂 전처리된 데이터 저장: {output_path}")

        summary = {
            'total_posts': len(df),
            'date_range': f"{df['created_at'].min()} ~ {df['created_at'].max()}",
            'unique_authors': df['author'].nunique(),
            'avg_likes': df['likes'].mean(),
            'avg_retweets': df['retweets'].mean(),
            'avg_views': df['views'].mean()
        }

        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        return df
