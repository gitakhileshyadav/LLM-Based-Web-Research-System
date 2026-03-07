"LLM App with Web Reasearch Tool"

import asyncio
import streamlit as st
from duckduckgo_search import DDGS

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

if __name__=="__main__":
    asyncio.run(run())
