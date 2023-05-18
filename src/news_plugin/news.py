import os
import json
import copy
import concurrent.futures
from tqdm import tqdm

import tiktoken
import openai
from trafilatura import fetch_url, extract
from duckduckgo_search import ddg_news
from tenacity import retry, stop_after_attempt, wait_random_exponential
# from duckduckgo_search.utils import SESSION

openai.api_key = os.getenv('OPENAI_API_KEY')

MODELS_INFO = {
    'gpt-3.5-turbo': {'max_tokens': 4096, 'pricing': 0.002/1000, 'tokenizer': tiktoken.get_encoding("cl100k_base"), 'tokens_per_message': 5},
    'gpt-4': {'max_tokens': 4096, 'pricing': 0.03/1000, 'tokenizer': tiktoken.get_encoding("cl100k_base"), 'tokens_per_message': 5},
}

def split_text(text, max_tokens=500, overlap=0, model='gpt-3.5-turbo'):
    tokenizer = MODELS_INFO[model]['tokenizer']
    tokens = tokenizer.encode(text)
    sid = 0
    splitted = []
    while True:
        if sid + overlap >= len(tokens):
            break
        eid = min(sid+max_tokens, len(tokens))
        splitted.append(tokenizer.decode(tokens[sid:eid]))
        sid = eid - overlap
    return splitted

def truncate_messages(messages, system_prompt="", model='gpt-3.5-turbo', n_response_tokens=500, keep_last=True):
    max_tokens = MODELS_INFO[model]['max_tokens']
    n_tokens_per_message = MODELS_INFO[model]['tokens_per_message']
    tokenizer = MODELS_INFO[model]['tokenizer']
    n_used_tokens = 3 + n_response_tokens
    n_used_tokens += n_tokens_per_message + len(tokenizer.encode(system_prompt))
    iterator = range(len(messages))
    if keep_last: 
        iterator = reversed(iterator)
    for i in iterator:
        message = messages[i]
        n_used_tokens += n_tokens_per_message
        if n_used_tokens >= max_tokens:
            messages = messages[i+1:] if keep_last else messages[:i]
            print('Messages Truncated')
            break
        content_tokens = tokenizer.encode(message['content'])
        n_content_tokens = len(content_tokens)
        n_used_tokens += n_content_tokens
        if n_used_tokens >= max_tokens:
            truncated_content_tokens = content_tokens[n_used_tokens-max_tokens:] if keep_last else content_tokens[:max_tokens-n_used_tokens]
            other_messages = messages[i+1:] if keep_last else messages[:i]
            messages = [{'role': message['role'], 'content': tokenizer.decode(truncated_content_tokens)}] + other_messages
            print('Messages Truncated')
            break
    return messages


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_chatgpt_response(messages:list, system_prompt="", model='gpt-3.5-turbo', temperature=0.5, keep_last=True):
    messages = copy.deepcopy(messages)
    messages = truncate_messages(messages, system_prompt, model, keep_last=keep_last)
    messages = [{"role": "system", "content": system_prompt}]+messages
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    response = dict(completion.choices[0].message)
    response['dollars_spent'] = completion['usage']['total_tokens'] * MODELS_INFO[model]['pricing']
    return response


class Agent:
    def __init__(
            self,
            model='gpt-3.5-turbo',
            n_urls=10,
            chunk_size=3000,
            chunk_overlap=10,
            proxy=None,
        ):
        # if 'NEWS_PROXY' in os.environ:
        #     proxy = os.getenv('NEWS_PROXY')
        #     SESSION.proxies = {
        #         "http": proxy,
        #         "https": proxy
        #     }
        if 'NEWS_MODEL' in os.environ:
            model = os.getenv('NEWS_MODEL')
        if 'NEWS_N_URLS' in os.environ:
            n_urls = int(os.getenv('NEWS_N_URLS'))
        if 'NEWS_CHUNK_SIZE' in os.environ:
            chunk_size = int(os.getenv('NEWS_CHUNK_SIZE'))
        if 'NEWS_CHUNK_OVERLAP' in os.environ:
            chunk_overlap = int(os.getenv('NEWS_CHUNK_OVERLAP'))

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
        urls = [x['url'] for x in search_results]
        # urls = self.get_relevant_urls(search_results)
        return urls
    
    def get_relevant_urls(self, search_results):
        prompt = ""
        for i, item in enumerate(search_results):
            prompt += f"""
```
Id: {i}
Title: {item['title']}
```"""
        prompt += f"""Given above search results, group search results based on the specific event. Respond in JSON format:
[
    {{"specific_event": "<event>", "ids": [<id1>, <id2>, ...]}},
    ...
]
"""
        response = get_chatgpt_response([{'role': 'user', 'content': prompt}], model=self.model, temperature=0)['content']
        relevent_ids = [x['ids'][0] for x in json.loads(response)]
        relevent_urls = [search_results[i]['url'] for i in relevent_ids]
        return relevent_urls
    
    def summarize_chunk(self, chunk, goal):
        prompt = f"""```
        {chunk}
        ```
        Summarize above content concisely, focusing on information potentially related to goal "{goal}".
        Summary:"""
        summary = get_chatgpt_response([{'role': 'user', 'content': prompt}], model=self.model, temperature=0)['content']
        return summary

    def summarize_doc(self, doc, goal):
        chunks = split_text(doc, max_tokens=self.chunk_size, overlap=self.chunk_overlap)
        def summarize_chunk_(chunk):
            return self.summarize_chunk(chunk, goal)
        while len(chunks) > 1:
            with concurrent.futures.ThreadPoolExecutor(len(chunks)) as executor:
                summaries = executor.map(summarize_chunk_, chunks)
            summary = "\n".join(summaries)
            chunks = split_text(summary, max_tokens=self.chunk_size, overlap=self.chunk_overlap)
        summary = summarize_chunk_(chunks[0])        
        return summary

    def generate_data(self, titles, summaries, goal):
        prompt = f""""""
        for i, (title, summary) in enumerate(zip(titles, summaries)):
            prompt += f"""
```
Id: {i+1}
Title: {title}
Summary: {summary}
```
"""
        prompt += f"""Concisely extract key information that is potentially related to goal "{goal}" from the above articles. Merge information from multiple articles if related. Respond in following JSON format:
[
    {{
        "key_information": "<key information 1>",
        "reference_ids": [<id 1>, <id 2>, ...]
    }}
    ...
]
"""
        data = get_chatgpt_response([{'role': 'user', 'content': prompt}], model=self.model, temperature=0)['content']
        data = json.loads(data)
        return data
    
    def data2report(self, data, urls):
        report = "\n".join([f"- {item['key_information']} {item['reference_ids']}" for item in data])
        used_reference_ids = set()
        for item in data:
            used_reference_ids = used_reference_ids.union(set(item['reference_ids']))
        report += "\nReferences\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(urls) if (i+1) in used_reference_ids])
        return report
    
    def __call__(self, keyword, goal):
        urls = self.get_urls(keyword)
        doc_dicts = [self.get_doc_from_url(url) for url in urls]
        doc_dicts = [doc_dict for doc_dict in doc_dicts if doc_dict is not None]
        titles = [doc_dict['title'] for doc_dict in doc_dicts]
        docs = [doc_dict['text'] for doc_dict in doc_dicts]
        def summarize_doc_(doc):
            return self.summarize_doc(doc, goal)
        with concurrent.futures.ThreadPoolExecutor(len(docs)) as executor:
            summaries = list(executor.map(summarize_doc_, docs))
        data = self.generate_data(titles, summaries, goal)
        report = self.data2report(data, urls)
        return report