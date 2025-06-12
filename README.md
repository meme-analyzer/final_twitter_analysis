🧬 밈 수명 주기 분석 (Twitter 기반)

트위터 데이터를 기반으로 밈(meme)의 생성, 확산, 쇠퇴 등 **생애주기 전반을 분석**하는 프로젝트입니다.  
Selenium을 사용한 크롤링부터 전처리, 시각화, 생존분석 및 클러스터링까지 자동 파이프라인 구축 완료.

---

📌 주요 기능

- 🔍 **트위터 데이터 수집** (Selenium 기반 크롤러)
- 🧹 **텍스트 전처리 및 임베딩 처리** (SentenceTransformer 활용)
- 📈 **밈 확산 시각화** (시간, 워드클라우드 등)
- 🧠 **밈 군집화 및 생애주기 클러스터링 분석**
- 📊 **생존 분석을 통한 밈 지속력 평가**
- 📦 **전체 파이프라인 자동 실행** (`run_pipeline_twitter.py`)

---

### 프로젝트 구조

```
meme_lifecycle_analysis/
├── data/
│   ├── raw/                # 원본 데이터
│   └── processed/          # 전처리된 데이터
├── src/
│   ├── collectors/         # 데이터 수집 모듈
│   ├── preprocessors/      # 데이터 전처리 모듈
│   ├── analyzers/          # 분석 모듈
│   └── visualizers/        # 시각화 모듈
├── results/
│   ├── figures/            # 그래프 이미지
│   └── reports/            # 분석 보고서
├── config/                 # 설정 파일
├── notebooks/              # Jupyter 노트북
└── run_pipeline_snsname/   # SNS별 전체 파이프라인 실행 스크립트
