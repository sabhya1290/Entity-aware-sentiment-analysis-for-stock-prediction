
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time

# page = 1
# base_url = f"https://www.bbc.com/search?q=stock+market&page={page}&edgeauth=eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJrZXkiOiAiZmFzdGx5LXVyaS10b2tlbi0xIiwiZXhwIjogMTc3MDE0NTg0OSwibmJmIjogMTc3MDE0NTQ4OSwicmVxdWVzdHVyaSI6ICIlMkZzZWFyY2glM0ZxJTNEc3RvY2slMkJtYXJrZXQifQ._o5gvE8k4SKYLTQ4zvVCrLfZJtjvstPFYugjliQfThM"


# data = {
#     "Headline": [],
#     "Time": []
# }

# while True:

#     url=base_url.format(page)
#     response = requests.get(url)

#     if response.status_code != 200:
#         print(f"No more pages after page {page-1}.")
#         break

#     print(f"Scraping page {page}...")

#     soup= BeautifulSoup(response.text,"html.parser")
#     classes= soup.find_all(class_="sc-cdecfb63-0 cJcHVD")

#     for c in classes:
    
#         headline = c.find(class_ = "sc-feaf8701-3 bhndtK")
#         data["Headline"].append(headline.string)

#         t = c.find(class_ = "sc-1907e52a-1 bZuSaP")
#         data["Time"].append(t.string)
    
#     time.sleep(2)
#     page+=1
    
#     if page == 3:  
#         print("scrapping is done!")
#         break

# df= pd.DataFrame.from_dict(data)
# df.to_csv("scrapping/stockNewsByBBC.csv",index=False)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {
    "User-Agent": "Mozilla/5.0"
}

data = {
    "text": [],
    "source": [],
    "time": []
}

for page in range(1, 30): 
    print(f"Scraping BBC page {page}...")
    # time.sleep(2)

    url = f"https://www.bbc.com/search?q=stock+market&page={page}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        break

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all(class_="sc-cdecfb63-0 cJcHVD")

    for article in articles:
        headline = article.find(class_ = "sc-feaf8701-3 bhndtK")
        time_tag = article.find(class_ = "sc-1907e52a-1 bZuSaP")

        if headline:
            data["text"].append(headline.get_text(strip=True))
            data["source"].append("BBC")
            data["time"].append(time_tag.get_text(strip=True) if time_tag else "NA")

bbc_df = pd.DataFrame(data)
print(bbc_df.head())
bbc_df.to_csv("scrapping/stockNewsByBBC.csv", index=False)
