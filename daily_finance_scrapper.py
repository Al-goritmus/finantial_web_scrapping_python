import requests
from bs4 import BeautifulSoup
import pandas as pd

headers= {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0'
}

url= f"https://stockanalysis.com/trending/"

xlwriter = pd.ExcelWriter('trending_top_stocks.xlsx', engine='xlsxwriter')

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
print(soup)
df = pd.read_html(str(soup), attrs={'id': 'main-table'})[0]
df.to_excel(xlwriter, sheet_name='trending_today', index=False)

xlwriter.save()



