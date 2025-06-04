import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# .env 환경 변수
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD")

# 루트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 경로
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# 결과물 경로
FIGURES_DIR = os.path.join("results", "figures")
REPORTS_DIR = os.path.join("results", "reports")

# 밈 목록
TARGET_MEMES = [
    "Italian brainrot",
    "chill guy",
    "나니가스키"
]

# 수집 설정
MAX_TWEETS_PER_MEME = 1000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

# 필요한 디렉토리 자동 생성
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)
