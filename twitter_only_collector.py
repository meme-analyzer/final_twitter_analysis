#!/usr/bin/env python3
"""
트위터 밈 수명 주기 분석 프로젝트 - 수집 전용 실행 파일
"""

import argparse
import time
from datetime import datetime
import sys
import os

# 현재 파일 기준으로 상위 디렉토리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils import create_directories
from src.collectors.selenium_twitter_collector import SeleniumTwitterCollector
from src.utils import create_directories
from config.config import TARGET_MEMES, RAW_DATA_DIR

def collect_twitter_data(meme_name):
    """Twitter에서 밈 데이터 수집"""
    print(f"\n=== Twitter에서 '{meme_name}' 데이터 수집 시작 ===")
    try:
        collector = SeleniumTwitterCollector(save_dir=RAW_DATA_DIR)
        posts = collector.search_posts(meme_name, max_posts=1000)
        collector.save_posts(posts, meme_name.replace(" ", "_"))
        collector.close()
        print(f"✓ {len(posts)}개의 트윗 수집 완료")
        return True
    except Exception as e:
        print(f"✗ Twitter 수집 실패: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Twitter 밈 데이터 수집 전용 실행기')
    parser.add_argument('--meme', type=str, help='수집할 밈 이름')
    parser.add_argument('--test', action='store_true', help='테스트 모드 (첫 번째 밈만 수집)')
    args = parser.parse_args()

    create_directories()

    if args.meme:
        memes_to_collect = [args.meme]
    elif args.test:
        memes_to_collect = [TARGET_MEMES[0]]
    else:
        memes_to_collect = TARGET_MEMES

    print("=== 트위터 밈 데이터 수집기 ===")
    print(f"수집 대상: {', '.join(memes_to_collect)}")
    print(f"시작 시간: {datetime.now()}\n")

    for meme in memes_to_collect:
        print(f"{'='*40}\n수집: {meme}\n{'='*40}")
        collect_twitter_data(meme)
        time.sleep(3)

    print(f"\n=== 수집 완료 ===")
    print(f"종료 시간: {datetime.now()}")

if __name__ == "__main__":
    main()
