'''
pip install -U duckduckgo_search
pip install -U langchain
'''
text="谷歌的股价"
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
search.run(text)
from duckduckgo_search import DDGS

with DDGS() as ddgs:
    keywords = text
    ddgs_news_gen = ddgs.news(
        keywords,
        region="wt-wt",
        safesearch="Off",
        timelimit="m",
    )
    for r in ddgs_news_gen:
        print(r)
