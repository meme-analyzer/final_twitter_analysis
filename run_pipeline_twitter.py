import argparse
import sys
import time
import glob
import os
import pandas as pd
from datetime import datetime

from src.collectors.selenium_twitter_collector import SeleniumTwitterCollector
from src.preprocessors.selenium_twitter_preprocessor import SeleniumTwitterPreprocessor
from src.visualizers.selenium_twitter_visualizer import SeleniumTwitterVisualizer
from src.analyzers.selenium_twitter_lifecycle_analyzer import SeleniumTwitterLifecycleAnalyzer
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def run_collection(meme_name):
    print(f"\n{'='*50}")
    print(f"1단계: Twitter 데이터 수집 - {meme_name}")
    print(f"{'='*50}")

    try:
        collector = SeleniumTwitterCollector(save_dir=RAW_DATA_DIR)
        posts = collector.search_posts(meme_name, max_posts=1000)

        if not posts:
            print("⚠ 게시물 수집 실패 또는 0건")
            return

        collector.save_posts(posts, meme_name.replace(" ", "_"))
        collector.close()
        print("✓ 수집 완료!")
    except Exception as e:
        print(f"Twitter 수집 실패: {e}")

def run_preprocessing(meme_name):
    print(f"\n{'='*50}")
    print(f"2단계: 데이터 전처리")
    print(f"{'='*50}")

    pattern = f"twitter_{meme_name.replace(' ', '_').lower()}_*.csv"
    files = glob.glob(os.path.join(RAW_DATA_DIR, pattern))
    if not files:
        print("✓ 전처리할 데이터 없음")
        return None

    latest_file = max(files, key=os.path.getctime)
    df_raw = pd.read_csv(latest_file)

    if df_raw.empty:
        print("⚠ CSV 파일이 비어 있음. 전처리 중단.")
        return None

    preprocessor = SeleniumTwitterPreprocessor()
    df_processed = preprocessor.preprocess(df_raw)

    processed_filename = f"processed_twitter_{meme_name.replace(' ', '_').lower()}.csv"
    df_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, processed_filename), index=False)
    print(f"✓ 전처리 완료: {processed_filename}")

    return processed_filename

def run_visualization(processed_filename, meme_name):
    print(f"\n{'='*50}")
    print(f"3단계: 시각화 생성")
    print(f"{'='*50}")

    visualizer = SeleniumTwitterVisualizer()
    filepath = os.path.join(PROCESSED_DATA_DIR, processed_filename)
    df = pd.read_csv(filepath)
    df['created_at'] = pd.to_datetime(df['created_at'])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    visualizer.plot_lifecycle_curve(df, meme_name.replace(' ', '_').lower())
    visualizer.plot_engagement_distribution(df, meme_name.replace(' ', '_').lower())
    visualizer.plot_time_series(df, meme_name.replace(' ', '_').lower())
    visualizer.plot_activity_heatmap(df, meme_name.replace(' ', '_').lower())
    visualizer.plot_wordcloud(df, meme_name.replace(' ', '_').lower())
    print("✓ 시각화 완료!")

def run_analysis(processed_filename, meme_name):
    print(f"\n{'='*50}")
    print(f"4단계: 수명 주기 분석")
    print(f"{'='*50}")

    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, processed_filename))
    df['date'] = pd.to_datetime(df['date'])

    analyzer = SeleniumTwitterLifecycleAnalyzer(save_dir=os.path.join("results", "reports"))
    metrics, growth, decline = analyzer.analyze(df, meme_name)
    analyzer.generate_text_report(meme_name, metrics, growth, decline)
    print("✓ 분석 및 보고서 생성 완료")

def main():
    parser = argparse.ArgumentParser(description="Twitter 밈 수명 주기 분석 파이프라인")
    parser.add_argument('--meme', type=str, default='chill guy', help='분석할 밈 이름')
    parser.add_argument('--skip-collection', action='store_true', help='수집 단계 생략')
    args = parser.parse_args()

    meme_name = args.meme
    print(f"\n{'='*60}")
    print(f"Twitter Meme Lifecycle 분석 시작")
    print(f"분석 대상: {meme_name}")
    print(f"시작 시간: {datetime.now()}")
    print(f"{'='*60}")

    try:
        if not args.skip_collection:
            run_collection(meme_name)
            time.sleep(1)

        processed = run_preprocessing(meme_name)
        if processed:
            time.sleep(1)
            run_visualization(processed, meme_name)
            time.sleep(1)
            run_analysis(processed, meme_name)

        print(f"\n{'='*60}")
        print("파이프라인 종료")
        print(f"종료 시간: {datetime.now()}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"[오류] 실행 중 문제 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
