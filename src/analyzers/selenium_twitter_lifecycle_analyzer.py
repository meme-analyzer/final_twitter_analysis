import os
import pandas as pd
from datetime import datetime

class SeleniumTwitterLifecycleAnalyzer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def analyze(self, df, meme_name):
        """
        밈 수명 주기 분석: 총량 통계, 성장기/쇠퇴기 탐지 + 비율 기반 지표 추가
        """
        print("\n📊 === 밈 분석 시작 ===")

        if 'date' not in df.columns:
            print("[경고] 'date' 컬럼이 없어 분석을 수행할 수 없습니다.")
            return {}, {}, {}

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        if df.empty:
            print("[경고] 유효한 날짜 데이터가 없어 분석을 수행할 수 없습니다.")
            return {}, {}, {}

        # 총량 통계 및 비율 지표
        df['like_rate'] = df['likes'] / df['views'].replace(0, pd.NA)
        df['retweet_rate'] = df['retweets'] / df['views'].replace(0, pd.NA)

        metrics = {
            'total_posts': len(df),
            'unique_authors': df['author'].nunique(),
            'date_range': f"{df['date'].min().date()} ~ {df['date'].max().date()}",
            'duration_days': (df['date'].max() - df['date'].min()).days + 1,
            'avg_likes': df['likes'].mean(),
            'avg_retweets': df['retweets'].mean(),
            'avg_views': df['views'].mean() if 'views' in df.columns else 0,
            'total_engagement': df['engagement_score'].sum(),
            'like_rate': df['like_rate'].mean(skipna=True),
            'retweet_rate': df['retweet_rate'].mean(skipna=True)
        }

        # 성장기: 일별 트윗 수 최댓값이 있는 날
        daily_counts = df.groupby('date').size()
        if daily_counts.empty:
            growth_phase = {'start_date': None, 'end_date': None, 'duration_days': 0}
        else:
            growth_peak_date = daily_counts.idxmax()
            growth_phase = {
                'start_date': growth_peak_date.date(),
                'end_date': growth_peak_date.date(),
                'duration_days': 1
            }

        # 쇠퇴기: 마지막 날짜
        decline_date = df['date'].max()
        decline_phase = {
            'start_date': decline_date.date(),
            'end_date': decline_date.date(),
            'duration_days': 1
        }

        print(f"📈 성장기: {growth_phase['start_date']}")
        print(f"📉 쇠퇴기: {decline_phase['start_date']}")

        return metrics, growth_phase, decline_phase

    def generate_text_report(self, meme_name, metrics, growth_phase=None, decline_phase=None):
        """
        분석 결과를 텍스트 리포트 파일로 저장
        """
        report_path = os.path.join(self.save_dir, f'{meme_name}_report.txt')
        print(f"\n📝 리포트 생성 중: {report_path}")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("MEME ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Meme: {meme_name}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # 1. 기본 통계
                f.write("1. BASIC STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Posts        : {metrics.get('total_posts', 'N/A')}\n")
                f.write(f"Unique Authors     : {metrics.get('unique_authors', 'N/A')}\n")
                f.write(f"Date Range         : {metrics.get('date_range', 'N/A')}\n")
                f.write(f"Duration           : {metrics.get('duration_days', 'N/A')} days\n")

                # 2. 반응 분석
                f.write("\n2. ENGAGEMENT\n")
                f.write("-" * 30 + "\n")
                f.write(f"Avg Likes          : {metrics.get('avg_likes', 0):.2f}\n")
                f.write(f"Avg Retweets       : {metrics.get('avg_retweets', 0):.2f}\n")
                f.write(f"Avg Views          : {metrics.get('avg_views', 0):.2f}\n")
                f.write(f"Total Engagement   : {metrics.get('total_engagement', 0)}\n")
                f.write(f"Like Rate (Likes/Views) : {metrics.get('like_rate', 0):.4f}\n")
                f.write(f"Retweet Rate (RT/Views) : {metrics.get('retweet_rate', 0):.4f}\n")

                # 3. 수명 주기
                f.write("\n3. LIFECYCLE PHASES\n")
                f.write("-" * 30 + "\n")
                if growth_phase and growth_phase.get("start_date"):
                    f.write(f"Growth Phase       : {growth_phase['start_date']} ~ {growth_phase['end_date']} ({growth_phase['duration_days']} days)\n")
                else:
                    f.write("Growth Phase       : N/A\n")

                if decline_phase and decline_phase.get("start_date"):
                    f.write(f"Decline Phase      : {decline_phase['start_date']} ~ {decline_phase['end_date']} ({decline_phase['duration_days']} days)\n")
                else:
                    f.write("Decline Phase      : N/A\n")

            print(f"✅ 분석 리포트 저장 완료: {report_path}")
            return report_path

        except Exception as e:
            print(f"[에러] 리포트 생성 실패: {e}")
            return None
