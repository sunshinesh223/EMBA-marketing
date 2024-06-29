import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import openpyxl


chromedriver_path = "/opt/homebrew/bin/chromedriver"
options = Options()
options.add_argument("--window-size=1920x1080")
options.add_argument("--verbose")

driver = webdriver.Chrome(options=options)

def scroll_to_element(element):
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
    time.sleep(1)


base_filepath = "./Data"

companys = [{"company_name":"colorline", "review_website":"https://no.trustpilot.com/review/www.colorline.no?languages=all"},
            {"company_name":"AIDA", "review_website":"https://no.trustpilot.com/review/www.aida.de?languages=all"},
            {"company_name":"DFDS", "review_website":"https://no.trustpilot.com/review/dfds.com?languages=all"},
            {"company_name":"Stena Line","review_website":"https://no.trustpilot.com/review/stenaline.com?languages=all"}]


for company in companys:
    company_name = company["company_name"]
    review_website = company["review_website"]
    driver.get(review_website)

    time.sleep(random.choice(range(2, 4)))

    all_data = []
    # Accept the cookies
    try:
        cookie_button = driver.find_element('xpath', "//button[contains(text(), 'Nei takk')]")
        cookie_button.click()
        time.sleep(2)  # Wait for the cookies consent to be processed
    except Exception as e:
        print("Cookies consent button not found or could not be clicked.", e)

    # Get the number of pages
    try:
        num_pages_elem = driver.find_element('xpath', "//a[@aria-label='Neste side']/preceding-sibling::a[1]")
        num_pages = int(num_pages_elem.text)
    except Exception as e:
        print("Could not determine the number of pages.", e)
        num_pages = 1  # Default to 1 page if not found

    for page_num in range(num_pages):
        print(page_num, len(all_data))
        for i in range(24):
            i = i + 4
            data = {}
            try:
                xpath_body = f"/html/body/div[1]/div/div/main/div/div[4]/section/div[{i}]/article/div/section/div[2]/p[1]"
                elem = driver.find_element('xpath', xpath_body)
                data["body_text"] = elem.text

                xpath_date = f"/html/body/div[1]/div/div/main/div/div[4]/section/div[{i}]/article/div/section/div[2]/p[2]"
                elem = driver.find_element('xpath', xpath_date)
                data["review_datetime_raw"] = elem.text

                xpath_title = f"/html/body/div[1]/div/div/main/div/div[4]/section/div[{i}]/article/div/section/div[2]/a/h2"
                elem = driver.find_element('xpath', xpath_title)
                data["review_title"] = elem.text

                xpath_star_rating = f"/html/body/div[1]/div/div/main/div/div[4]/section/div[{i}]/article/div/section/div[1]/div[1]/img"
                elem = driver.find_element('xpath', xpath_star_rating)
                data["rating_raw"] = elem.get_attribute("alt")

                #print(i, len(all_data), data)
                all_data.append(data)
            except Exception as e:
                pass
                #print(i, e)

        # Go to the next page
        if page_num < num_pages - 1 and page_num < 300:  # Avoid clicking next on the last page
            next_page_xpath = "//a[@data-pagination-button-next-link='true' and @aria-label='Neste side']"
            elem = driver.find_element('xpath', next_page_xpath)
            time.sleep(1)
            scroll_to_element(elem)
            time.sleep(3)
            elem.click()
            time.sleep(5)
            time.sleep(random.choice(range(10)))

    num_reviews = len(all_data)
    df = pd.DataFrame.from_records(all_data)
    print(f"Number of unique reviews for {company_name}: {df.body_text.nunique()}")

    # Save to pickle file
    filepath_pickle = os.path.join(base_filepath, f"{company_name}_trustpilot_{num_reviews}_reviews_raw.pkl")
    df.to_pickle(filepath_pickle)
    print(f"Pickle file saved: {filepath_pickle}")

    # Save to Excel file
    filepath_excel = os.path.join(base_filepath, f"{company_name}_trustpilot_{num_reviews}_reviews_raw.xlsx")
    df.to_excel(filepath_excel, index=False)
    print(f"Excel file saved: {filepath_excel}")

driver.quit()