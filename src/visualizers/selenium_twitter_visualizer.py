import os
import sys
import platform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from datetime import datetime
from wordcloud import WordCloud

# ✅ 경로 설정 - 상위 디렉토리(config 모듈 경로 등) 불러오기
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DATA_DIR, FIGURES_DIR

# ✅ 한글 폰트 설정 함수 (run_pipeline_twitter.py와 같은 경로에 HMKMG.TTF 존재해야 함)
def set_korean_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치 (src/visualizers)
    font_path = os.path.join(base_dir, "..", "..", "HMKMG.TTF")  # 봉코드/HMKMG.TTF

    if not os.path.exists(font_path):
        raise FileNotFoundError(f"[❌] 폰트 파일이 없습니다: {font_path}")

    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    print(f"[✅ 폰트 적용 완료] {font_prop.get_name()}")

# ✅ 한글 폰트 적용
set_korean_font()

# ✅ 트위터 시각화 클래스
class SeleniumTwitterVisualizer:
    def __init__(self):
        self.processed_dir = PROCESSED_DATA_DIR
        self.output_dir = FIGURES_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # 시각화 스타일 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def load_processed_data(self, filename):
        filepath = os.path.join(self.processed_dir, filename)
        return pd.read_csv(filepath)

    # ✅ 생명주기 곡선 (일별 게시물 + 누적 게시물)
    def plot_lifecycle_curve(self, df, meme_name):
        df['date'] = pd.to_datetime(df['date'])
        daily_posts = df.groupby('date').size()
        daily_posts_ma = daily_posts.rolling(window=7, min_periods=1).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 일별
        ax1.plot(daily_posts.index, daily_posts.values, alpha=0.3, label='일일 게시물 수')
        ax1.plot(daily_posts_ma.index, daily_posts_ma.values, linewidth=2, label='7일 이동 평균')
        ax1.set_title(f'{meme_name} 밈 생명주기')
        ax1.set_xlabel('날짜')
        ax1.set_ylabel('게시물 수')
        ax1.legend()
        ax1.grid(True)

        # 누적
        cumulative_posts = daily_posts.cumsum()
        ax2.plot(cumulative_posts.index, cumulative_posts.values, color='green')
        ax2.fill_between(cumulative_posts.index, cumulative_posts.values, alpha=0.3, color='green')
        ax2.set_title('누적 게시물 수')
        ax2.set_xlabel('날짜')
        ax2.set_ylabel('누적 게시물')
        ax2.grid(True)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{meme_name}_lifecycle_curve.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[저장] 생명주기 곡선: {path}")

    # ✅ 참여 점수 분포 히스토그램
    def plot_engagement_distribution(self, df, meme_name):
        plt.figure(figsize=(8, 4))
        sns.histplot(df['engagement_score'], bins=30, kde=True)
        plt.title(f"{meme_name} - 참여 점수 분포")
        plt.xlabel("참여 점수")
        plt.ylabel("빈도")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{meme_name}_engagement_score.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[저장] 참여 점수 분포: {path}")

    # ✅ 언급량 시계열
    def plot_time_series(self, df, meme_name):
        df['date'] = pd.to_datetime(df['date'])
        plt.figure(figsize=(10, 5))
        df.groupby('date').size().plot(kind='line', marker='o')
        plt.title(f"밈 언급 시계열 - {meme_name}")
        plt.xlabel("날짜")
        plt.ylabel("언급 수")
        plt.grid(True)
        filepath = os.path.join(self.output_dir, f"{meme_name}_timeseries.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[저장] 시간 시계열: {filepath}")

    # ✅ 클러스터링 시각화 (x, y, cluster 컬럼 필요)
    def plot_clusters(self, df, meme_name):
        required_cols = {'x', 'y', 'cluster'}
        if not required_cols.issubset(df.columns):
            print(f"[경고] 클러스터링 시각화를 위한 컬럼 {required_cols}이 누락되었습니다.")
            return

        if df[['x', 'y']].isnull().any().any():
            print("[경고] x 또는 y 값에 NaN이 포함되어 있어 클러스터링 시각화를 건너뜁니다.")
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='Set2', s=60, edgecolor='k')
        plt.title(f"{meme_name} - 클러스터 시각화")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{meme_name}_clusters.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[저장] 클러스터링 시각화: {path}")

    # ✅ 생존 분석 (lifelines 이용, created_at ~ last_seen_at)
    def plot_survival_curve(self, df, meme_name):
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            print("[에러] lifelines 패키지가 설치되어 있지 않습니다. 생존 분석을 건너뜁니다.")
            return

        required_cols = {'created_at', 'last_seen_at'}
        if not required_cols.issubset(df.columns):
            print(f"[경고] 생존 분석에 필요한 컬럼 {required_cols}이 누락되었습니다.")
            return

        try:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['last_seen_at'] = pd.to_datetime(df['last_seen_at'], errors='coerce')
            df = df.dropna(subset=['created_at', 'last_seen_at'])

            df['duration'] = (df['last_seen_at'] - df['created_at']).dt.days
            df = df[df['duration'] >= 0]
            df['event_observed'] = 1

            if df.empty:
                print("[경고] 유효한 생존 분석 데이터가 없어 시각화를 건너뜁니다.")
                return

            kmf = KaplanMeierFitter()
            kmf.fit(df['duration'], event_observed=df['event_observed'])

            plt.figure(figsize=(8, 5))
            kmf.plot_survival_function()
            plt.title(f"{meme_name} - 생존 분석")
            plt.xlabel("일수")
            plt.ylabel("생존 확률")
            plt.grid(True)
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"{meme_name}_survival.png")
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"[저장] 생존 분석: {path}")

        except Exception as e:
            print(f"[에러] 생존 분석 중 예외 발생: {e}")

    # ✅ 시간/요일별 히트맵
    def plot_activity_heatmap(self, df, meme_name):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df.dropna(subset=['created_at'])
        df['hour'] = df['created_at'].dt.hour
        df['day'] = df['created_at'].dt.dayofweek  # 0=월 ~ 6=일

        pivot = df.pivot_table(index='day', columns='hour', values='text', aggfunc='count').fillna(0)

        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", cbar=True)
        plt.title(f"{meme_name} - 활동 히트맵 (요일 vs 시간)")
        plt.xlabel("시간")
        plt.ylabel("요일 (0=월, 6=일)")
        path = os.path.join(self.output_dir, f"{meme_name}_heatmap.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[저장] 히트맵 시각화: {path}")

    # ✅ 워드클라우드 생성 (text_clean 컬럼 기준)
    def plot_wordcloud(self, df, meme_name):
        if 'text_clean' not in df.columns:
            print("[경고] 워드클라우드를 위한 text_clean 컬럼이 없습니다.")
            return

        text_data = ' '.join(df['text_clean'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Dark2').generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{meme_name}_wordcloud.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[저장] 워드클라우드: {path}")
