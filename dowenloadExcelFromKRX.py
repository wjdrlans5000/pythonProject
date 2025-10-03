import os
import time
import shutil
import datetime
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC  # <<<<< 이 줄을 추가해야 합니다!


def download_krx_excel(stock_code, stock_name, download_path, chromedriver_path):
    # 크롬 옵션 설정 (자동 다운로드)
    options = Options()
    prefs = {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)

    # 크롬 드라이버 실행 (Service 객체 사용)
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    try:
        url = "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020203#"
        driver.get(url)
        wait = WebDriverWait(driver, 30)
        print("✅ KRX 페이지 접속 성공!")

        # 1. '종목시세' 메뉴 클릭
        stock_price_menu = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#jsMdiMenu > div.lnb_tree > ul > li:nth-child(1) > ul > li.sel > div > div.lnb_tree_box > ul > li.sel.on > ul > li:nth-child(1) > a"))
        )

        stock_price_menu.click()
        print("1. '종목시세' 메뉴 클릭 성공!")

        menu = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-menu-id='MDC0201020103']"))
        )
        menu.click()
        print("개별종목 시세 추이 클릭 성공!")

        # 종목명 검색창 클릭
        # search_btn = wait.until(
        #     EC.element_to_be_clickable((By.CSS_SELECTOR, "#btnisuCd_finder_stkisu0_1"))
        # )
        # search_btn.click()

        # 종목 검색창 뜨면 입력
        input_box = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input#tboxisuCd_finder_stkisu0_1"))
        )
        # 기존 값 삭제
        input_box.clear()
        input_box.send_keys(stock_code)
        input_box.send_keys(Keys.ENTER)
        time.sleep(5) # 검색어 입력 후 잠시 대기

        # 팝업 닫기 버튼 클릭 (예: X 버튼)
        # try:
        #     close_btn = wait.until(
        #         EC.element_to_be_clickable((By.CSS_SELECTOR, "#jsLayer_finder_stkisu0_1 > div.ui-dialog-titlebar.ui-corner-all.ui-widget-header.ui-helper-clearfix.ui-draggable-handle > button > span.ui-button-icon.ui-icon.ui-icon-closethick"))
        #     )
        #     close_btn.click()
        # except:
        #     pass  # 팝업이 없으면 무시
        # time.sleep(3)

        # 검색 버튼 클릭
        # stock_search = wait.until(
        #     EC.presence_of_element_located((By.CSS_SELECTOR, "#searchBtn__finder_stkisu0_1"))
        # )
        # stock_search.click()
        # time.sleep(5)

        # 종목 리스트 로딩 대기 첫번째 선택
        # if not stock_name == 'NAVER' :
        #     firstTd = wait.until(
        #         EC.presence_of_element_located((By.CSS_SELECTOR, "#jsGrid__finder_stkisu0_1 > tbody > tr:nth-child(1) > td.tal.pl20"))
        #     )
        #     firstTd.click()
        #     time.sleep(3)

        # 기간 → 1년 선택
        year = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#MDCSTAT017_FORM > div.search_tb > div > table > tbody > tr:nth-child(2) > td > div > div > button.cal-btn-range1y"))
        )
        year.click()
        time.sleep(1) # 기간 설정 적용 대기

        search = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a#jsSearchButton"))
        )
        # search.click()
        driver.execute_script("arguments[0].click();", search)
        time.sleep(1) # 기간 설정 적용 대기


        # 엑셀 다운로드 버튼 클릭
        # 다운로드 버튼이 항상 마지막에 뜨는 것은 아닐 수 있으므로 좀 더 긴 대기가 필요할 수 있습니다.
        download_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#UNIT-WRAP1 > div > p:nth-child(2) > button.CI-MDI-UNIT-DOWNLOAD"))
        )
        download_btn.click()

        # 엑셀 아이콘 클릭
        excel_icon = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#ui-id-3 > div > div:nth-child(1) > a"))
        )
        excel_icon.click()
        time.sleep(3) # 다운로드 완료 대기
        print("다운로드 성공!")

    finally:
        driver.quit()

    # ... (다운로드 파일 이동/정리 로직은 그대로 유지) ...
    # 최신 다운로드 파일 찾기
    files = [os.path.join(download_path, f) for f in os.listdir(download_path)]
    files = [f for f in files if f.endswith(".xls") or f.endswith(".xlsx")]
    latest_file = max(files, key=os.path.getctime)

    # 새 파일명 지정 (예: 삼성전자_20250924.xlsx)
    today = datetime.datetime.today().strftime("%Y%m%d")
    new_filename = f"{stock_name}_{today}.xlsx"

    # 최종 저장 경로
    final_path = os.path.join(fr"C:\Users\PC\IdeaProjects\pythonProject2\data\{today}", new_filename)

    # 날짜 폴더 없으면 생성
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    shutil.move(latest_file, final_path)
    print(f"✅ 다운로드 완료: {final_path}")
    return final_path


if __name__ == "__main__":
    stock_list = [
        "005930/삼성전자", "000660/SK하이닉스", "373220/LG에너지솔루션", "035420/NAVER", "015760/한국전력",
        "302440/SK바이오사이언스", "247540/에코프로비엠", "005490/POSCO홀딩스", "005380/현대차", "105560/KB금융",
        "086790/하나금융지주", "034020/두산에너빌리티", "012450/한화에어로스페이스", "267250/HD현대", "402340/SK스퀘어",
        "006400/삼성SDI", "051910/LG화학", "035720/카카오", "196170/알테오젠", "068270/셀트리온",
        "329180/HD현대중공업", "042660/한화오션"
    ]
    download_folder = r"C:\Users\PC\Downloads"
    chromedriver_path = r"C:\Users\PC\Downloads\chromedriver-win64\chromedriver.exe"

    for stock_item  in stock_list:
        try:
            stock_code, stock_name = stock_item.split("/")  # 종목코드, 종목명 분리
            print(f"\n--- {stock_name} ({stock_code}) 종목 데이터 다운로드 시작 ---")
            download_krx_excel(stock_code,stock_name, download_folder, chromedriver_path)  # 여기서는 코드만 전달
            print(f"--- {stock_name} ({stock_code}) 종목 다운로드 완료 ---")
        except Exception as e:
            print(f"❌ {stock_name} ({stock_code}) 다운로드 중 오류 발생: {e}")
            continue  # 오류 발생 시 다음 종목으로 넘어감