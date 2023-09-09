import requests
import json
from bs4 import BeautifulSoup

url = "https://www.binance.com/en/futures/crypto-heatmap/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

script_tag = soup.find("script", id="__APP_DATA", type="application/json")
script_content = script_tag.string

# Extract the "topSearchedSymbols" data
data = json.loads(script_content)
top_searched_symbols = data['pageData']['redux']['markets']['topSearchedSymbols']['um']

# Filter the data to get the top 50 rank symbols
filtered_symbols = [item["symbol"] for item in top_searched_symbols if item["rank"] <= 50]

print(filtered_symbols)