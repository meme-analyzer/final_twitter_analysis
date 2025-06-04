import os
import csv
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SeleniumTwitterCollector:
    def __init__(self, save_dir, show_browser=True):
        # 저장 디렉토리 생성
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # .env에서 트위터 계정 정보 로딩
        load_dotenv()
        self.username = os.getenv("TWITTER_USERNAME")
        self.password = os.getenv("TWITTER_PASSWORD")

        # 크롬 드라이버 옵션 설정
        options = Options()
        if not show_browser:
            options.add_argument("--headless")
        options.add_argument("--window-size=1400,1000")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--lang=ko-KR")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # 크롬 드라이버 실행
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        print("🌐 브라우저 초기화 및 실행 완료")

    def load_cookies(self):
        # 쿠키 파일을 로드하여 자동 로그인 수행
        import pickle
        cookie_path = os.path.join("config", "twitter_cookies.pkl")
        if not os.path.exists(cookie_path):
            raise FileNotFoundError("❌ 쿠키 파일이 없습니다. 먼저 save_twitter_cookies.py로 로그인 후 쿠키 저장하세요.")

        print("🍪 트위터 접속 중...")
        self.driver.get("https://twitter.com")
        time.sleep(2)

        print("🔑 쿠키 로딩 중...")
        with open(cookie_path, "rb") as f:
            cookies = pickle.load(f)
            for cookie in cookies:
                self.driver.add_cookie(cookie)

        print("🔄 페이지 새로고침 중...")
        self.driver.refresh()
        time.sleep(3)
        print("✅ 로그인 완료!")

    def extract_engagement_counts(self, card):
        # 좋아요, 리트윗, 댓글 수, 조회수 추출 함수
        likes = '0'
        retweets = '0'
        replies = '0'
        views = '0'

        try:
            container = card.find_element(By.CSS_SELECTOR, 'div[aria-label*="likes"]')
            aria_label = container.get_attribute('aria-label')
            print("=" * 50)
            print("aria-label 내용:", aria_label)
            print("=" * 50)
            match = re.search(r'(\d+(?:,\d+)?) replies?, (\d+(?:,\d+)?) reposts?, (\d+(?:,\d+)?) likes?,?.*?(\d+(?:,\d+)?) views?', aria_label)
            if match:
                replies, retweets, likes, views = match.groups()
        except Exception as e:
            print(f"[디버그] aria-label 파싱 실패: {e}")

        return likes, retweets, replies, views

        def search_posts(self, keyword, max_posts=None):  # max_posts는 이제 옵션
        print(f"🔍 '{keyword}' 검색 시작...")
        self.load_cookies()

        from urllib.parse import quote  # 검색어 인코딩
        encoded_keyword = quote(keyword)
        self.driver.get(f"https://twitter.com/search?q={encoded_keyword}&src=typed_query&f=live")
        time.sleep(3)

        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="tweet"]'))
            )
            print("✅ 트윗 요소 로딩 완료")
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            return []

        posts = []
        seen_urls = set()
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            cards = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            print(f"🔄 현재 감지된 트윗 수: {len(cards)}")
            new_count = 0

            for card in cards:
                try:
                    # 트윗 URL 중복 방지
                    url_elem = card.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]')
                    url = url_elem.get_attribute('href')
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # 텍스트 추출
                    text_elems = card.find_elements(By.CSS_SELECTOR, 'div[data-testid="tweetText"] span')
                    text = ' '.join([e.text for e in text_elems if e.text.strip()])

                    # 사용자명 추출
                    username = "unknown"
                    username_elems = card.find_elements(By.CSS_SELECTOR, 'div[data-testid="User-Name"] span')
                    for elem in username_elems:
                        if elem.text.strip() and '@' not in elem.text:
                            username = elem.text.strip()
                            break

                    # 게시 시간 추출
                    try:
                        timestamp = card.find_element(By.TAG_NAME, 'time').get_attribute('datetime')
                    except:
                        timestamp = datetime.now().isoformat()

                    # 좋아요/리트윗/댓글/조회수 추출
                    likes, retweets, replies, views = self.extract_engagement_counts(card)

                    # 해시태그 추출
                    hashtags = ','.join(re.findall(r'#\w+', text))

                    # 데이터 저장
                    post = {
                        'author': username,
                        'text': text,
                        'hashtags': hashtags,
                        'likes': likes,
                        'retweets': retweets,
                        'replies': replies,
                        'views': views,
                        'created_at': timestamp,
                        'url': url
                    }
                    posts.append(post)
                    new_count += 1

                    print(f"📥 {username}: ❤️{likes} 🔁{retweets} 💬{replies} 👁️{views}")

                    if max_posts and len(posts) >= max_posts:
                        print("📦 최대 수집 개수 도달로 종료")
                        return posts
                except:
                    continue

            print(f"✅ 새로 수집된 트윗 수: {new_count}")

            # 새 게시물이 하나도 없으면 종료
            if new_count == 0:
                print("🛑 더 이상 새로운 트윗이 로드되지 않음. 종료합니다.")
                break

            # 스크롤 다운
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            # 페이지 높이 변화 확인
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("📉 페이지 높이에 변화 없음. 종료합니다.")
                break
            last_height = new_height

        print(f"🎉 총 {len(posts)}개 트윗 수집 완료")
        return posts


    def save_posts(self, posts, meme_name):
        # 수집한 게시물 CSV로 저장
        if not posts:
            print("⚠️ 저장할 게시물이 없습니다.")
            return
        filename = f"twitter_{meme_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=posts[0].keys())
            writer.writeheader()
            writer.writerows(posts)
        print(f"✅ 저장 완료: {filepath}")

    def close(self):
        print("🔚 브라우저를 종료합니다...")
        time.sleep(2)
        self.driver.quit()
        print("✅ 종료 완료")
