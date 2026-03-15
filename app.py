
"LLM App with Web Reasearch Tool"

import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import streamlit as st


from duckduckgo_search import DDGS

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult

async def crawl_webpages(urls:list[str], prompt:str)-> CrawlResult:
    bm25_filter=BM25ContentFilter(user_query=prompt, bm25_threshold=1.2)
    md_generator=DefaultMarkdownGenerator(content_filter=bm25_filter)

    crawler_config=CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav","footer","header","form","img","a"],
        only_text=True,
        excluded_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) AppleWebKit/537.36 ",
        page_timeout=20000, # in ns:20 seconds
    )
    browser_config=BrowserConfig(headless=True, text_mode=True, light_mode=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results=await crawler.arun_many(urls, config=crawler_config)
        return results

def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls=[]
    for url in urls:
        try:
            robots_url=f"{urlparse(url).scheme}//{urlparse(url).netloc}/robots.txt"
            rp=RobotFileParser(robots_url)
            rp.read()

            if rp.can_fetch("*",url):
                allowed_urls.append(url)
        except Exception:
            #If robots.txt is missing or there is any error assume url is allowed
            allowed_urls.append(url)
    return allowed_urls

def get_web_urls(search_term: str, numresults: int =5,)-> list[str]:
    try:
        discard_url=["youtube.com","bitanica.com","vimeo.com"]
        for url in discard_url:
            search_term += f"-site{url}"
        
        st.write(search_term)

        results=DDGS().text(search_term,max_results=numresults)
        st.write(results)
        return check_robots_txt(results)
    
    except Exception as e:
        error_msg=("Failed to fetch the result from the web ",str(e))
        print(error_msg)
        st.write(error_msg)
        st.stop()


async def run():
    #Title of the page
    st.set_page_config(page_title="LLM with Web Search")

    #Heading on the page
    st.header("LLM Web Search")
    prompt=st.text_area(
        label="Put your query here..",
        placeholder="Add you query..", #Translucent text inside the box
        label_visibility="hidden",
    )
    is_web_search=st.toggle("Eanble web Search", value=False,key="enable_web_search")
    go=st.button(
        "GO",
    )
    if prompt and go:
        if is_web_search:
            web_urls=get_web_urls(search_term=prompt)
            if not web_urls:
                st.write("No result found")
                st.stop()

if __name__=="__main__":
    asyncio.run(run())
