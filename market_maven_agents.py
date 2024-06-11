from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, timedelta
import openai
import google.generativeai as genai
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect
import ast

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_AI_KEY = os.getenv("NEWSAPI_AI_KEY")
NEWSAPI_ORG_KEY = os.getenv("NEWSAPI_ORG_KEY")

def gpt_summaraizer(news_text):
    prompt_template = PromptTemplate.from_template(
        "Task : Summarize the data and always provide just the results in English\n"
        "Text: '{text}'"
        "Response: result of the above task")
    prompt = prompt_template.format(text=news_text)
    # Use openai.Completion.create for text generation

    response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
    message_content = response.choices[0].message.content

    return message_content

def gemini_summaraizer(news_text):
    prompt_template = PromptTemplate.from_template(
        "Task : consice only english news and always display only the results\n"
        "Text: '{text}'"
                )
    prompt = prompt_template.format(text=news_text)
    # Specify the Gemini model
    gen_model = genai.GenerativeModel("gemini-pro")
    genai.configure(api_key=GEMINI_API_KEY)
    response = gen_model.generate_content(prompt)
    return response.text

def nlp(news_text):
    print("OPENAI_API_KEY:",OPENAI_API_KEY) 
    prompt_template = PromptTemplate.from_template(
    """
    **Input:** {text}
    Given a sentence, identify the main entity or concept and return python list containing keyword from the sentence.  
    **Refer to the following  Examples:**
        if the Input is 'give me the updates regarding nba' then Output should be python list containing 'nba'
        if the Input is 'what is the latest news on apple' then Output should be python list containing 'apple'
        if the Input is 'which is best gemini or chatgpt' then output should be python list containing 'gemini','chatgpt'
    **Additional Notes:** 
    * display only the Output field
    * Focus on identifying the most relevant entity for the sentence and display in python lit format.        
    * Response should be of type string it should be a python list containg strings.""" 
    # """
    # **Input:** {text}
    # Given a sentence, identify the main entity or concept and return python list containing keyword from the sentence.  
    # **Refer to the following Examples:**
    #     if the Input is 'give me the updates regarding nba' then Output should be ['nba']
    #     if the Input is 'what is the latest news on apple' then Output should be ['apple']
    #     if the Input is 'which is best gemini or chatgpt' then Output should be ['gemini','chatgpt']
    # **Additional Notes:** 
    # * Response should be of type list containing strings.
    # * Display only the response   
    # """

    )   
    prompt = prompt_template.format(text=news_text)
    # Use openai.Completion.create for text generation
  
    response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
    message_content = response.choices[0].message.content
    print(message_content)         
    actual_list = ast.literal_eval(message_content)
    return actual_list  
    # prompt = prompt_template.format(text=news_text)
    # # Specify the Gemini model
    # gen_model = genai.GenerativeModel("gemini-pro")
    # genai.configure(api_key=GEMINI_API_KEY)
    # response = gen_model.generate_content(prompt)
    # return response.text

def analytics(news_text,query):
    prompt_template = PromptTemplate.from_template(
        """
        User Question:{query} 
        News Updates:{news_txt}
        you are an ai assistant who will help out the user with some insightful information related to the user query using the provided News Updates, Assume the provided News Updates are the latest news updates.
        Understand the user question first
        if the {query} is not answerable using the above mentioned new updates then just generate a response that Ensure that the 
        response will highlight and correlate the interrelations between the above given News Updates, Provide insightful commentary or analysis where 
        applicable.
        if the {query} is answerable using the above mentioned news updates then generate a comprehensive response to user questionand Ensure that the response 
        directly addresses the user's query while incorporating relevant information from the news updates. If there are 
        interrelations between the updates, highlight and correlate them. Provide insightful commentary or analysis where 
        applicable.""")
    prompt = prompt_template.format(news_txt=news_text,query=query)
    # Use openai.Completion.create for text generation

    response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
    message_content = response.choices[0].message.content

    return message_content
    # prompt = prompt_template.format(news_txt=news_text,query=query)
    # # Specify the Gemini model
    # gen_model = genai.GenerativeModel("gemini-pro")
    # genai.configure(api_key=GEMINI_API_KEY)
    # response = gen_model.generate_content(prompt)
    # return response.text

def newsapi1(query,model):
    global NEWSAPI_ORG_KEY
    # print(NEWSAPI_ORG_KEY)
    # Get today's date
    today = datetime.today()
    yesterday = today - timedelta(days=1) 
    # Format the date as YYYY-MM-DD
    from_date = yesterday.strftime('%Y-%m-%d')
    to_date=today.strftime('%Y-%m-%d')
    # print(formatted_date)
    # query='python'
    print('query',query)
    url=f'https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=popularity&apiKey={NEWSAPI_ORG_KEY}'
    # url=str(url)
    r=requests.get(url)
    news=json.loads(r.text)
    # print("news",news)
    articles=news['articles']
    # print('articles:',articles)
    if len(articles)>=10:
        main_articles=[]
        for article in articles[:5]:
            # print(article)
            # print(article['title'])
            # print(article['url'])
            # print("******************")
            if article['content'] != None and article['title']!='null' :
                if article['title']!="[Removed]":
                    if is_english(article['title']):
                        main_article=newsapi1_selected_info(article)
                        main_articles.append(main_article)
    else:
        main_articles=[]
        for article in articles:
            if article['content'] != None and article['title']!='null' :
                if article['title']!="[Removed]":
                    if is_english(article['title']):
                        main_article=newsapi1_selected_info(article)
                        main_articles.append(main_article)
    return main_articles

def new_one(query,model):
    print("entered")
    global NEWSAPI_AI_KEY
    url=f'https://eventregistry.org/api/v1/article/getArticles?action=getArticles&lang=eng&keyword={query}&articlesPage=1&articlesCount=5&articlesSortBy=date&articlesSortByAsc=false&articlesArticleBodyLen=-1&resultType=articles&dataType%5B%5D=news&dataType%5B%5D=pr&apiKey={NEWSAPI_AI_KEY}&forceMaxDataTimeWindow=31'
    r=requests.get(url)
    news=json.loads(r.text)
    articles=news['articles']
    articles_results=articles['results']
    # print('articles_results',articles_results)
    main_articles=''
    for article in articles_results:
            title=article['title']
            desc=article['body']
            urll=article['url']
            if is_english(title):
                # main_article=newsapi2_selected_info(article)
                # main_articles.append(main_article)
                main_articles+=f"Title: {title}\n+Description:{desc}\n+Url:{urll}\n"
    # print(main_articles)
    return main_articles

def newsapi2(query,model):
    global NEWSAPI_AI_KEY
    url=f'https://eventregistry.org/api/v1/article/getArticles?action=getArticles&lang=eng&keyword={query}&articlesPage=1&articlesCount=5&articlesSortBy=date&articlesSortByAsc=false&articlesArticleBodyLen=-1&resultType=articles&dataType%5B%5D=news&dataType%5B%5D=pr&apiKey={NEWSAPI_AI_KEY}&forceMaxDataTimeWindow=31'
    r=requests.get(url)
    news=json.loads(r.text)
    articles=news['articles']
    articles_results=articles['results']
    # print('articles_results',articles_results)
    main_articles=[]
    for article in articles_results:
            title=article['title']
            if is_english(title):
                main_article=newsapi2_selected_info(article)
                main_articles.append(main_article)
    return main_articles

def newsapi2_selected_info(article):
    selected_keys = ['title', 'body', 'url']
    selected_info = {key: article[key] for key in selected_keys if key in article}
    return selected_info

def newsapi1_selected_info(article):
    selected_keys = ['title', 'content', 'url']
    selected_info = {key: article[key] for key in selected_keys if key in article}
    return selected_info

def is_english(text):
    try:
        # Detect the language of the text
        language = detect(text)
        # Check if the detected language is English
        return language == 'en'
    except LangDetectException:
        # Handle the case where language detection fails
        return False