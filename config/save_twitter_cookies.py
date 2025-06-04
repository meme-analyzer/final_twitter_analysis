from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pickle
import os

# 저장할 경로
COOKIE_PATH = "config/twitter_cookies.pkl"

options = Options()
options.add_experimental_option("detach", True)  # 창 자동 종료 막기
options.add_argument("--lang=ko-KR")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 트위터 로그인 페이지 열기
driver.get("https://twitter.com/login")

print("\n⏳ 로그인 후 30초 기다립니다. 직접 로그인하세요.")
time.sleep(30)

# 쿠키 저장
cookies = driver.get_cookies()
with open("config/twitter_cookies.pkl", "wb") as f:
    pickle.dump(driver.get_cookies(), f)
print(f"✅ 쿠키 저장 완료: {COOKIE_PATH}")
driver.quit()
