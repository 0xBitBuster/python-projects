from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['META', 'AMZN']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    # Use random user agent to not get flagged
    ua = UserAgent()
    response = requests.get(url, headers={'User-Agent': ua.random})

    # Scrape data
    html = BeautifulSoup(response.content, 'html.parser')
    news_table = html.find(id="news-table")
    news_tables[ticker] = news_table

parsed_data = []
for ticker, news_table in news_tables.items():
    parent_date = None

    for row in news_table.findAll('tr'):
        title = row.a.text.strip()
        date_data = row.td.text.strip().split(' ')

        if len(date_data) == 1:
            date = parent_date
            time = date_data[0]
        else:
            date = date_data[0] if date_data[0] != "Today" else datetime.now().strftime("%b-%d-%y")
            time = date_data[1]
            parent_date = date

        parsed_data.append([ticker, date, time, title])

vader = SentimentIntensityAnalyzer()
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
df['compound'] = df['title'].apply(lambda t: vader.polarity_scores(t)['compound'])
df['date'] = pd.to_datetime(df.date, format="%b-%d-%y").dt.date

mean_df = df.groupby(['ticker', 'date']).sum().unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()
mean_df.plot(kind="bar")
plt.show()
