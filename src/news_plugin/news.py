import numpy as np
from trafilatura import fetch_url, extract
import concurrent.futures
import secrets
import time
from tqdm import tqdm
from dotenv import load_dotenv
from duckduckgo_search import ddg_news
import json
from utils import *
load_dotenv()


class Agent:
    def __init__(
            self,
            model='gpt-3.5-turbo',
            n_urls=10,
            chunk_size=3000,
            chunk_overlap=10,
        ):
        self.model = model
        self.n_urls = n_urls
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_doc_from_url(self, url):
        downloaded = fetch_url(url)
        doc = extract(downloaded, favor_recall=True, output_format="json")
        if doc is None:
            return None
        return json.loads(doc)

    def get_urls(self, query):      
        search_results = ddg_news(query, safesearch='off', max_results=self.n_urls)
        urls = [item['url'] for item in search_results]
        return urls
    
    def summarize_chunk(self, chunk):
        prompt = f"""Summarize following content delimeted by triple backticks.
        ```
        {chunk}
        ```
        Summary:"""
        summary = get_chatgpt_response([{'role': 'user', 'content': prompt}], model=self.model, temperature=0)['content']
        return summary

    def summarize_doc(self, doc):
        chunks = split_text(doc, max_tokens=self.chunk_size, overlap=self.chunk_overlap)
        while len(chunks) > 1:
            with concurrent.futures.ThreadPoolExecutor(len(chunks)) as executor:
                summaries = executor.map(self.summarize_chunk, chunks)
            summary = "\n".join(summaries)
            chunks = split_text(summary, max_tokens=self.chunk_size, overlap=self.chunk_overlap)
        summary = self.summarize_chunk(chunks[0])        
        return summary

    def generate_data(self, titles, summaries):
        prompt = f""""""
        for i, (title, summary) in enumerate(zip(titles, summaries)):
            prompt += f"""
```
Title: {title}
Summary: {summary}
Id: {i+1}
```
"""
        prompt += """Concisely extract key informations from the above articles delimeted by triple backticks. Merge information from multiple articles if related. Respond in following JSON format:
[
    {
        "key_information": "<key information 1>",
        "reference_ids": ["<id 1>", "<id 2>", ...]
    }
    ...
]
"""
        data = get_chatgpt_response([{'role': 'user', 'content': prompt}], model=self.model, temperature=0)['content']
        data = json.loads(correct_json(data))
        return data
    
    def data2report(self, data, urls):
        report = "\n".join([f"- {item['key_information']} {item['reference_ids']}" for item in data])
        report += "\nReferences\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(urls)])
        return report
    
    def __call__(self, keyword):
        urls = self.get_urls(keyword)
        doc_dicts = [self.get_doc_from_url(url) for url in urls]
        doc_dicts = [doc_dict for doc_dict in doc_dicts if doc_dict is not None]
        titles = [doc_dict['title'] for doc_dict in doc_dicts]
        docs = [doc_dict['text'] for doc_dict in doc_dicts]
        with concurrent.futures.ThreadPoolExecutor(len(docs)) as executor:
            summaries = list(executor.map(self.summarize_doc, docs))
        data = self.generate_data(titles, summaries)
        report = self.data2report(data, urls)
        return report