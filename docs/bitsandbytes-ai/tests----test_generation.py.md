# `bitsandbytes\tests\test_generation.py`

```py
# 导入所需的模块
from itertools import product
import math
import pytest
import torch
import transformers
from transformers import (
  AutoModelForCausalLM,
  BitsAndBytesConfig,
)

from tests.helpers import TRUE_FALSE, describe_dtype, id_formatter

# 定义一个返回4位配置的函数
def get_4bit_config():
  return BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
  )

# 定义一个根据配置获取模型和分词器的函数
def get_model_and_tokenizer(config):
    model_name_or_path, quant_type = config
    bnb_config = get_4bit_config()
    # 根据量化类型设置配置
    if quant_type == '16bit':
        bnb_config.load_in_4bit = False
    else:
        bnb_config.bnb_4bit_quant_type= quant_type
    # 从预训练模型加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
        quantization_config=bnb_config,
        max_memory={0:'48GB'},
        device_map='auto',
        torch_dtype=torch.bfloat16
        ).eval()
    # 从预训练模型加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

# 定义一个生成用于评估的提示函数
def get_prompt_for_generation_eval(text, add_roles=True):
    description = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    if add_roles:
        prompt = f'{description} ### Human: {text} ### Assistant:'
    else:
        prompt = f'{description} {text}'
    return prompt

# 定义一个生成函数
def generate(model, tokenizer, text, generation_config, prompt_func=get_prompt_for_generation_eval):
    text = prompt_func(text)
    # 使用分词器对文本进行编码
    inputs = tokenizer(text, return_tensors="pt").to('cuda:0')
    # 生成文本
    outputs = model.generate(inputs=inputs['input_ids'], generation_config=generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 定义模型和数据类型列表
models = ['huggyllama/llama-7b', 'bigscience/bloom-1b7']
dtypes = ['nf4', 'fp4']
# 定义一个 pytest 的 fixture，用于生成模型和分词器的参数组合
@pytest.fixture(scope='session', params=product(models, dtypes))
def model_and_tokenizer(request):
    # 调用函数获取模型和分词器
    model, tokenizer = get_model_and_tokenizer(request.param)
    # 生成参数组合
    yield request.param, model, tokenizer
    # 删除模型对象

# 使用 pytest.mark.parametrize 注册测试参数，用于测试 pi 的生成
@pytest.mark.parametrize("DQ", TRUE_FALSE, ids=id_formatter("dq"))
@pytest.mark.parametrize("inference_kernel", TRUE_FALSE, ids=id_formatter("inference_kernel"))
@pytest.mark.parametrize("dtype", [torch.float16], ids=describe_dtype)
@pytest.mark.slow
def test_pi(requires_cuda, model_and_tokenizer, inference_kernel, DQ, dtype):
    # 获取 fixture 中的参数
    fixture_config, model, tokenizer = model_and_tokenizer

    # 配置生成文本的参数
    generation_config = transformers.GenerationConfig(
        max_new_tokens=20,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    generation_config.max_new_tokens = 20

    # 设置生成文本的初始内容
    n_cases = 6
    text = '3.14159'
    
    # 如果模型配置中有量化配置，则设置量化参数
    if hasattr(model.config, 'quantization_config'):
        model.config.quantization_config.bnb_4bit_compute_dtype = dtype
        model.config.quantization_config.bnb_4bit_use_double_quant = DQ

    # 根据推理内核是否开启，设置输入文本
    if not inference_kernel:
        text = [text]*n_cases
    inputs = tokenizer(text, return_tensors="pt").to('cuda:0')
    x = inputs['input_ids']
    outputs = []
    
    # 根据推理内核是否开启，生成文本
    if inference_kernel:
        for i in range(n_cases):
            output = model.generate(x, generation_config=generation_config)
            textout = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(textout)
    else:
        outputs = model.generate(x, generation_config=generation_config)
        outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # 断言生成的文本是否符合预期
    assert len(outputs) == n_cases
    failure_count = 0
    for i in range(n_cases):
        if not outputs[i][:len(str(math.pi))] == str(math.pi):
            failure_count += 1
    # 根据 fixture_config 的第一个元素是否为 'huggyllama/llama-7b' 来确定 failure_max 的值
    failure_max = (2 if fixture_config[0] == 'huggyllama/llama-7b' else 4)
    # 如果 failure_count 大于 failure_max，则执行以下操作
    if failure_count > failure_max:
        # 打印数学常数 pi
        print(math.pi)
        # 遍历 outputs 列表中的元素，并打印出来
        for out in outputs:
            print(out)
        # 抛出 ValueError 异常，包含失败次数和总用例数的信息
        raise ValueError(f'Failure count: {failure_count}/{n_cases}')
```