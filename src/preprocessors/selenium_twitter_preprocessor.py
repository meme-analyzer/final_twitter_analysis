import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class SeleniumTwitterPreprocessor:
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def load_twitter_data(self, filename):
        filepath = os.path.join(self.raw_data_dir, filename)
        df = pd.read_csv(filepath)
        print(f"📥 데이터 로드 완료: {len(df)}개 게시물")
        return df

    def preprocess(self, df):
        print("\n🧹=== 트위터 데이터 전처리 시작 ===")

        # 날짜, 시간, 요일 추출
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek

        # 결측값 처리
        df['text'] = df['text'].fillna('')
        df['author'] = df['author'].fillna('[deleted]')
        df['likes'] = pd.to_numeric(df.get('likes', 0), errors='coerce').fillna(0).astype(int)
        df['retweets'] = pd.to_numeric(df.get('retweets', 0), errors='coerce').fillna(0).astype(int)
        df['views'] = pd.to_numeric(df.get('views', 0), errors='coerce').fillna(0).astype(int)

        # 텍스트 클렌징
        df['text_clean'] = df['text'].apply(self.clean_text)

        # 참여 점수 계산
        df['engagement_score'] = df['likes'] + df['retweets'] * 2 + df['views'] * 0.1

        # 최신순 정렬
        df = df.sort_values(by='created_at', ascending=False).reset_index(drop=True)

        print(f"✅ 전처리 완료: {len(df)}개 게시물")
        return df

    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def analyze_temporal_patterns(self, df):
        print("\n⏱️=== 시간 패턴 분석 ===")
        daily_posts = df.groupby('date').size()
        hourly_dist = df['hour'].value_counts().sort_index()
        day_dist = df['day_of_week'].value_counts().sort_index()
        days = ['월', '화', '수', '목', '금', '토', '일']

        print(f"📆 데이터 기간: {daily_posts.index.min()} ~ {daily_posts.index.max()}")
        print(f"📈 일 평균 게시물 수: {daily_posts.mean():.2f}")
        print(f"⏰ 가장 활발한 시간대: {hourly_dist.idxmax()}시")
        print(f"🗓️ 가장 활발한 요일: {days[day_dist.idxmax()]}요일")

        return {
            'daily_posts': daily_posts,
            'hourly_dist': hourly_dist,
            'day_dist': day_dist
        }

    def perform_clustering(self, df, n_clusters=5):
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
        print("\n🔍=== 생존 분석용 마지막 등장 시점 추정 ===")
        grouped = df.groupby('text_clean')['created_at'].max().reset_index()
        grouped.columns = ['text_clean', 'last_seen_at']
        df = df.merge(grouped, on='text_clean', how='left')
        print("✅ last_seen_at 컬럼 생성 완료")
        return df

    def save_processed_data(self, df, output_filename):
        os.makedirs(self.processed_data_dir, exist_ok=True)
        output_path = os.path.join(self.processed_data_dir, output_filename)
        df.to_csv(output_path, index=False)
        print(f"💾 전처리된 데이터 저장: {output_path}")

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
