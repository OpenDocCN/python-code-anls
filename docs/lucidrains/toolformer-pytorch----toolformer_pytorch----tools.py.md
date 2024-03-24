# `.\lucidrains\toolformer-pytorch\toolformer_pytorch\tools.py`

```py
# 导入所需的库
import os

# 尝试导入所需的库，如果导入失败则输出错误信息并退出程序
try:
    # 从dotenv库中导入load_dotenv函数
    from dotenv import load_dotenv
    load_dotenv()

    # 导入requests、calendar、wolframalpha、datetime、AutoModelForSeq2SeqLM、AutoTokenizer、pow、truediv、mul、add、sub等库
    import requests
    import calendar
    import wolframalpha
    import datetime
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from operator import pow, truediv, mul, add, sub

    # 可选导入
    from googleapiclient.discovery import build

# 如果导入失败，则输出错误信息并退出程序
except ImportError:
    print('please run `pip install tools-requirements.txt` first at project directory')
    exit()

'''
Calendar

使用Python的datetime和calendar库来获取当前日期。

input - 无

output - 一个字符串，表示当前日期。
'''
def Calendar():
    # 获取当前时间
    now = datetime.datetime.now()
    # 返回当前日期的字符串表示
    return f'Today is {calendar.day_name[now.weekday()]}, {calendar.month_name[now.month]} {now.day}, {now.year}.'


'''
Wikipedia Search

使用ColBERTv2来检索维基百科文档。

input_query - 一个字符串，输入查询（例如"what is a dog?"）
k - 要检索的文档数量

output - 一个字符串列表，每个字符串是一个维基百科文档

改编自Stanford的DSP: https://github.com/stanfordnlp/dsp/
也可参考: https://github.com/lucabeetz/dsp
'''
class ColBERTv2:
    def __init__(self, url: str):
        self.url = url

    def __call__(self, query, k=10):
        topk = colbertv2_get_request(self.url, query, k)

        topk = [doc['text'] for doc in topk]
        return topk

# 发送ColBERTv2请求
def colbertv2_get_request(url: str, query: str, k: int):
    payload = {'query': query, 'k': k}
    res = requests.get(url, params=payload)

    topk = res.json()['topk'][:k]
    return topk

# 维基百科搜索函数
def WikiSearch(
    input_query: str,
    url: str = 'http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search',
    k: int = 10
):
    retrieval_model = ColBERTv2(url)
    output = retrieval_model(input_query, k)
    return output

'''
Machine Translation - NLLB-600M

使用HuggingFace的transformers库将输入查询翻译成英文。

input_query - 一个字符串，输入查询（例如"what is a dog?"）

output - 一个字符串，翻译后的输入查询。
'''
def MT(input_query: str, model_name: str = "facebook/nllb-200-distilled-600M"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer(input_query, return_tensors='pt')
    outputs = model.generate(
        **input_ids,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], 
        )
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return output


'''
Calculator

计算数学表达式的结果。

input_query - 一个字符串，输入的数学表达式（例如"400/1400"）

output - 一个浮点数，计算结果

改编自: https://levelup.gitconnected.com/3-ways-to-write-a-calculator-in-python-61642f2e4a9a 
'''
def Calculator(input_query: str):
    operators = {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv
        }
    if input_query.isdigit():
        return float(input_query)
    for c in operators.keys():
        left, operator, right = input_query.partition(c)
        if operator in operators:
            return round(operators[operator](Calculator(left), Calculator(right)), 2)


# 其他可选工具


'''
Wolfram Alpha Calculator

pip install wolframalpha

使用Wolfram Alpha API计算输入查询。

input_query - 一个字符串，输入查询（例如"what is 2 + 2?"）

output - 一个字符串，输入查询的答案

wolfarm_alpha_appid - 你的Wolfram Alpha API密钥
'''
def WolframAlphaCalculator(input_query: str):
    wolfram_alpha_appid = os.environ.get('WOLFRAM_ALPHA_APPID')
    wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
    res = wolfram_client.query(input_query)
    assumption = next(res.pods).text
    answer = next(res.results).text
    return f'Assumption: {assumption} \nAnswer: {answer}'


'''
Google Search

使用Google的自定义搜索API来检索Google搜索结果。

input_query - 要搜索的查询。
# The number of results to return for the Google Custom Search API
num_results - The number of results to return.
# Your Google API key for accessing Google Custom Search API
api_key - Your Google API key.
# Your Google Custom Search Engine ID for identifying the custom search engine
cse_id - Your Google Custom Search Engine ID.

# A function to perform a custom search using Google Custom Search API
# Returns a list of dictionaries, each dictionary representing a Google Search result
'''
def custom_search(query, api_key, cse_id, **kwargs):
    # Build a service object for the Google Custom Search API
    service = build("customsearch", "v1", developerKey=api_key)
    # Execute the search query and retrieve the results
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res['items']

# A function to perform a Google search using the custom_search function
def google_search(input_query: str, num_results: int = 10):
    # Retrieve Google API key and Custom Search Engine ID from environment variables
    api_key = os.environ.get('GOOGLE_API_KEY')
    cse_id = os.environ.get('GOOGLE_CSE_ID')

    metadata_results = []
    # Perform custom search using custom_search function
    results = custom_search(input_query, num=num_results, api_key=api_key, cse_id=cse_id)
    # Extract relevant metadata from search results
    for result in results:
        metadata_result = {
            "snippet": result["snippet"],
            "title": result["title"],
            "link": result["link"],
        }
        metadata_results.append(metadata_result)
    return metadata_results

'''
Bing Search

Uses Bing's Custom Search API to retrieve Bing Search results.

input_query: The query to search for.
bing_subscription_key: Your Bing API key.
num_results: The number of results to return.

output: A list of dictionaries, each dictionary is a Bing Search result
'''
# A function to retrieve Bing search results using Bing's Custom Search API
def _bing_search_results(
    search_term: str,
    bing_subscription_key: str,
    count: int,
    url: str = "https://api.bing.microsoft.com/v7.0/search"
):
    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {
        "q": search_term,
        "count": count,
        "textDecorations": True,
        "textFormat": "HTML",
    }
    # Make a GET request to Bing API to retrieve search results
    response = requests.get(
        url, headers=headers, params=params
    )
    response.raise_for_status()
    search_results = response.json()
    return search_results["webPages"]["value"]

# A function to perform a Bing search using the _bing_search_results function
def bing_search(
    input_query: str,
    num_results: int = 10
):
    # Retrieve Bing API key from environment variables
    bing_subscription_key = os.environ.get("BING_API_KEY")
    metadata_results = []
    # Perform Bing search using _bing_search_results function
    results = _bing_search_results(input_query, bing_subscription_key, count=num_results)
    # Extract relevant metadata from search results
    for result in results:
        metadata_result = {
            "snippet": result["snippet"],
            "title": result["name"],
            "link": result["url"],
        }
        metadata_results.append(metadata_result)
    return metadata_results

# Main function to demonstrate the usage of various search functions
if __name__ == '__main__':
 
    print(Calendar()) # Outputs a string, the current date

    print(Calculator('400/1400')) # For Optional Basic Calculator

    print(WikiSearch('What is a dog?')) # Outputs a list of strings, each string is a Wikipedia document

    print(MT("Un chien c'est quoi?")) # What is a dog?

    # Optional Tools

    print(WolframAlphaCalculator('What is 2 + 2?')) # 4

    print(google_search('What is a dog?')) 
    # Outputs a list of dictionaries, each dictionary is a Google Search result

    print(bing_search('What is a dog?')) 
    # Outputs a list of dictionaries, each dictionary is a Bing Search result
```