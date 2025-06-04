import os

def create_directories():
    """필요한 디렉토리들을 생성"""
    from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR

    directories = [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[디렉토리 확인/생성됨] {directory}")
