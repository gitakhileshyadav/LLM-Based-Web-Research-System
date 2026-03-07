"LLM App with Web Reasearch Tool"

import asyncio
import streamlit as st
from duckduckgo_search import DDGS

def get_web_urls(search_term: str, numresults: int =10,)-> list[str]:
    try:
        discard_url=["youtube.com","bitanica.com","vimeo.com"]
        for url in discard_url:
            search_term += f"-site{url}"
        
        st.write(search_term)

        results=DDGS().text(search_term,max_results=numresults)
        st.write(results)
        return results
    
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
