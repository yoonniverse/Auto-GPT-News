import json
import re
import tiktoken
import openai
import copy
import requests
import contextlib
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from dotenv import dotenv_values
env = dotenv_values(".env") 
openai.api_key = env['OPENAI_API_KEY']

MODELS_INFO = {
    'gpt-3.5-turbo': {'max_tokens': 4096, 'pricing': 0.002/1000, 'tokenizer': tiktoken.get_encoding("cl100k_base"), 'tokens_per_message': 5},
    'gpt-4': {'max_tokens': 4096, 'pricing': 0.03/1000, 'tokenizer': tiktoken.get_encoding("cl100k_base"), 'tokens_per_message': 5},
}

def count_tokens(text, model='gpt-3.5-turbo'):
    return len(MODELS_INFO[model]['tokenizer'].encode(text))

def truncate_text(text, n_tokens, model='gpt-3.5-turbo'):
    tokenizer = MODELS_INFO[model]['tokenizer']
    tokens = tokenizer.encode(text)
    tokens = tokens[:n_tokens]
    return tokenizer.decode(tokens)

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

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    Returns the number of tokens used by a list of messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

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
            break
        content_tokens = tokenizer.encode(message['content'])
        n_content_tokens = len(content_tokens)
        n_used_tokens += n_content_tokens
        if n_used_tokens >= max_tokens:
            truncated_content_tokens = content_tokens[n_used_tokens-max_tokens:] if keep_last else content_tokens[:max_tokens-n_used_tokens]
            other_messages = messages[i+1:] if keep_last else messages[:i]
            messages = [{'role': message['role'], 'content': tokenizer.decode(truncated_content_tokens)}] + other_messages
            break
    return messages

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_chatgpt_response(messages:list, system_prompt="", model='gpt-3.5-turbo', temperature=0.5, wrap_quote=False, keep_last=True, suffix=''):
    messages = copy.deepcopy(messages)
    if wrap_quote:
        messages = [{'role': message['role'], 'content': f'```{message["content"]}```'} for message in messages]   
    messages[-1]['content'] += suffix
    messages = truncate_messages(messages, system_prompt, model, keep_last=keep_last)
    messages = [{"role": "system", "content": system_prompt}]+messages
    if model == 'gpt-4':
        print(f'Using model gpt-4!')
        openai.api_type = "azure"
        openai.api_base = "https://upstage.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = env["OPENAI_API_KEY_AZURE"]
        completion = openai.ChatCompletion.create(
            engine=model,
            messages=messages,
            temperature=temperature,
        )
    else:
        openai.api_type = 'open_ai'
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    response = dict(completion.choices[0].message)
    response['dollars_spent'] = completion['usage']['total_tokens']*MODELS_INFO[model]['pricing']
    return response

def get_embedding(doc, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=doc,
        model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

def compress_query(messages):
    prompt = """You will be given chat messages between user and assistant delimeted by triple backticks.
Provide a goal for the assistant in following JSON format:
{
    "goal": "goal description",
}
Chat messages:
"""
    context = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
    prompt += f"```{context}```"
    response = get_chatgpt_response([{'role': 'user', 'content': prompt}], model='gpt-3.5-turbo', temperature=0, keep_last=True)['content']
    response = json.loads(correct_json(response))
    return response['goal']


def extract_char_position(error_message: str) -> int:
    """Extract the character position from the JSONDecodeError message.

    Args:
        error_message (str): The error message from the JSONDecodeError
          exception.

    Returns:
        int: The character position.
    """

    char_pattern = re.compile(r"\(char (\d+)\)")
    if match := char_pattern.search(error_message):
        return int(match[1])
    else:
        raise ValueError("Character position not found in the error message.")


def fix_invalid_escape(json_to_load: str, error_message: str) -> str:
    """Fix invalid escape sequences in JSON strings.

    Args:
        json_to_load (str): The JSON string.
        error_message (str): The error message from the JSONDecodeError
          exception.

    Returns:
        str: The JSON string with invalid escape sequences fixed.
    """
    while error_message.startswith("Invalid \\escape"):
        bad_escape_location = extract_char_position(error_message)
        json_to_load = (
            json_to_load[:bad_escape_location] + json_to_load[bad_escape_location + 1 :]
        )
        try:
            json.loads(json_to_load)
            return json_to_load
        except json.JSONDecodeError as e:
            error_message = str(e)
    return json_to_load


def balance_braces(json_string: str):
    """
    Balance the braces in a JSON string.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with braces balanced.
    """

    open_braces_count = json_string.count("{")
    close_braces_count = json_string.count("}")

    while open_braces_count > close_braces_count:
        json_string += "}"
        close_braces_count += 1

    while close_braces_count > open_braces_count:
        json_string = json_string.rstrip("}")
        close_braces_count -= 1

    with contextlib.suppress(json.JSONDecodeError):
        json.loads(json_string)
        return json_string


def add_quotes_to_property_names(json_string: str) -> str:
    """
    Add quotes to property names in a JSON string.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with quotes added to property names.
    """

    def replace_func(match: re.Match) -> str:
        return f'"{match[1]}":'

    property_name_pattern = re.compile(r"(\w+):")
    corrected_json_string = property_name_pattern.sub(replace_func, json_string)

    try:
        json.loads(corrected_json_string)
        return corrected_json_string
    except json.JSONDecodeError as e:
        raise e


def correct_json(json_to_load: str) -> str:
    """
    Correct common JSON errors.
    Args:
        json_to_load (str): The JSON string.
    """
    if json_to_load.startswith('"'):
        json_to_load = json_to_load[1:]
    if json_to_load.endswith('"'):
        json_to_load = json_to_load[:-1]
    try:
        json.loads(json_to_load)
        return json_to_load
    except json.JSONDecodeError as e:
        error_message = str(e)
        if error_message.startswith("Invalid \\escape"):
            json_to_load = fix_invalid_escape(json_to_load, error_message)
        if error_message.startswith(
            "Expecting property name enclosed in double quotes"
        ):
            json_to_load = add_quotes_to_property_names(json_to_load)
            try:
                json.loads(json_to_load)
                return json_to_load
            except json.JSONDecodeError as e:
                error_message = str(e)
        if balanced_str := balance_braces(json_to_load):
            return balanced_str
    return json_to_load