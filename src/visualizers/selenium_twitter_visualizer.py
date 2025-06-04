import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from collections import Counter


class SeleniumTwitterVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    # 1. 밈 게시물 일별 수 변화 (생애주기 곡선)
    def plot_daily_post_trend(self, df):
        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby('date').size()
        ma = daily.rolling(window=7, min_periods=1).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(daily.index, daily.values, alpha=0.4, label='Daily Count')
        plt.plot(ma.index, ma.values, label='7-Day MA', linewidth=2)
        plt.title("Daily Meme Post Trend")
        plt.xlabel("Date")
        plt.ylabel("Tweet Count")
        plt.legend()
        path = os.path.join(self.output_dir, "daily_post_trend.png")
        plt.savefig(path)
        plt.close()

    # 2. 참여 점수 분포 시각화
    def plot_engagement_distribution(self, df):
        plt.figure(figsize=(8, 4))
        sns.histplot(df['engagement_score'], bins=30, kde=True)
        plt.title("Engagement Score Distribution")
        plt.xlabel("Engagement Score")
        path = os.path.join(self.output_dir, "engagement_distribution.png")
        plt.savefig(path)
        plt.close()

    # 3. 요일-시간대별 트윗 활동 히트맵
    def plot_heatmap_by_day_hour(self, df):
        df['day_abbr'] = pd.to_datetime(df['date']).dt.day_name().str[:3].str.upper()
        pivot = df.pivot_table(index='day_abbr', columns='hour', values='text', aggfunc='count').fillna(0)
        order = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        pivot = pivot.reindex(order)

        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title("Tweet Activity Heatmap (Day vs Hour)")
        plt.xlabel("Hour")
        plt.ylabel("Day of Week (Abbr)")
        path = os.path.join(self.output_dir, "heatmap_day_hour.png")
        plt.savefig(path)
        plt.close()

    # 4. 텍스트 클렌징 기반 워드클라우드
    def plot_wordcloud(self, df):
        text = ' '.join(df['text_clean'].dropna())
        if os.name == 'nt':  # Windows
            font_path = "C:/Windows/Fonts/malgun.ttf"
        elif os.name == 'posix':  # macOS/Linux
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        else:
            print("[⚠️] 알 수 없는 운영체제입니다. 기본 폰트로 시도합니다.")
            font_path = None   

        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        path = os.path.join(self.output_dir, "wordcloud.png")

        plt.savefig(path)
        plt.close()

        
    # 5. 최다 해시태그 상위 N개 바 차트
    def plot_top_hashtags(self, df, top_n=20):
        # 한글 폰트 설정
        if os.name == 'nt':
            font_path = "C:/Windows/Fonts/malgun.ttf"
        else:
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    
        all_tags = df['hashtags'].dropna().tolist()
        flat_tags = [tag for tags in all_tags for tag in str(tags).split() if tag.startswith('#')]
        counter = Counter(flat_tags)
        common = counter.most_common(top_n)
        if not common:
            print("[경고] 해시태그가 충분하지 않아 시각화를 건너뜁니다.")
            return
        tags, counts = zip(*common)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(tags))
        plt.title("Top Hashtags")
        plt.xlabel("Count")
        path = os.path.join(self.output_dir, "top_hashtags.png")
        plt.savefig(path)
        plt.close()

    # 6. 좋아요 vs 조회수 산점도
    def plot_likes_vs_views(self, df):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='views', y='likes', data=df, alpha=0.6)
        plt.title("Likes vs Views")
        plt.xlabel("Views")
        plt.ylabel("Likes")
        path = os.path.join(self.output_dir, "likes_vs_views.png")
        plt.savefig(path)
        plt.close()

    # 7. 좋아요 vs 리트윗 산점도
    def plot_likes_vs_retweets(self, df):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='retweets', y='likes', data=df, alpha=0.6)
        plt.title("Likes vs Retweets")
        plt.xlabel("Retweets")
        plt.ylabel("Likes")
        path = os.path.join(self.output_dir, "likes_vs_retweets.png")
        plt.savefig(path)
        plt.close()

    # 8. 생존 분석 곡선 (Kaplan-Meier)
    def plot_survival_curve(self, df):
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            print("[에러] lifelines 패키지가 설치되어 있지 않습니다. 생존 분석을 건너뜁니다.")
            return

        if 'created_at' not in df.columns or 'last_seen_at' not in df.columns:
            print("[경고] 생존 분석에 필요한 컬럼이 없습니다.")
            return

        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['last_seen_at'] = pd.to_datetime(df['last_seen_at'], errors='coerce')
        df = df.dropna(subset=['created_at', 'last_seen_at'])
        df['duration'] = (df['last_seen_at'] - df['created_at']).dt.days
        df = df[df['duration'] >= 0]
        df['event_observed'] = 1

        if df.empty:
            print("[경고] 유효한 생존 데이터가 없어 시각화를 건너뜁니다.")
            return

        kmf = KaplanMeierFitter()
        kmf.fit(df['duration'], event_observed=df['event_observed'])

        plt.figure(figsize=(8, 5))
        kmf.plot_survival_function()
        plt.title("Survival Curve of Meme (Kaplan-Meier)")
        plt.xlabel("Days")
        plt.ylabel("Survival Probability")
        path = os.path.join(self.output_dir, "survival_curve.png")
        plt.savefig(path)
        plt.close()

    # 9. 좋아요 & 조회수 시간별 추이
    def plot_likes_views_trend(self, df):
        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby('date')[['likes', 'views']].sum()
        ma = daily.rolling(window=7, min_periods=1).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(daily.index, daily['likes'], alpha=0.3, label='Likes')
        plt.plot(ma.index, ma['likes'], label='Likes (7d MA)')
        plt.plot(daily.index, daily['views'], alpha=0.3, label='Views')
        plt.plot(ma.index, ma['views'], label='Views (7d MA)')
        plt.title("Likes & Views Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.legend()
        path = os.path.join(self.output_dir, "likes_views_trend.png")
        plt.savefig(path)
        plt.close()

    # 10. 리트윗 시간별 추이
    def plot_retweet_trend(self, df):
        df['date'] = pd.to_datetime(df['date'])
        daily_retweets = df.groupby('date')['retweets'].sum()
        ma = daily_retweets.rolling(window=7, min_periods=1).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(daily_retweets.index, daily_retweets.values, alpha=0.4, label='Daily Retweets')
        plt.plot(ma.index, ma.values, label='7-Day MA', linewidth=2)
        plt.title("Retweet Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Retweet Count")
        plt.legend()
        path = os.path.join(self.output_dir, "retweet_trend.png")
        plt.savefig(path)
        plt.close()

    # 11. 좋아요 비율 (Like Rate) 분포 시각화
    def plot_like_rate_distribution(self, df):
        df = df[df['views'] > 0].copy()
        df['like_rate'] = df['likes'] / df['views']
        df = df[df['like_rate'].between(0, 1)]

        plt.figure(figsize=(8, 4))
        sns.histplot(df['like_rate'], bins=30, kde=True)
        plt.title("Like Rate Distribution (Likes / Views)")
        plt.xlabel("Like Rate")
        path = os.path.join(self.output_dir, "like_rate_distribution.png")
        plt.savefig(path)
        plt.close()
