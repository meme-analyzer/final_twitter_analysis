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

📁 프로젝트 구조

📦 bongcode/
├── config/                   # 설정값 및 경로 정의
├── data/
│   ├── raw/                 # 수집된 원시 트위터 데이터
│   └── processed/           # 전처리된 CSV 파일
├── results/
│   ├── figures/             # 시각화 이미지 저장 폴더
│   └── reports/             # 분석 리포트 저장 폴더
├── src/
│   ├── collectors/          # 트위터 크롤러 (Selenium 기반)
│   ├── preprocessors/       # 전처리 및 임베딩 처리
│   ├── visualizers/         # 시각화 코드 (시간/감성/워드클라우드 등)
│   ├── analyzers/           # 생애주기 분석, 클러스터링, 생존분석 등
│   └── utils/               # 공통 유틸 함수
├── run_pipeline_twitter.py  # 전체 파이프라인 실행 스크립트
└── README.md                # 프로젝트 설명 문서
