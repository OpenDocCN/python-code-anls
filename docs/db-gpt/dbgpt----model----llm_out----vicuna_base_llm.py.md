# `.\DB-GPT-src\dbgpt\model\llm_out\vicuna_base_llm.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


@torch.inference_mode()
# 定义生成对话流的函数，基于指定的模型、分词器、参数和设备
def generate_stream(
    model, tokenizer, params, device, context_len=4096, stream_interval=2
):
    """Fork from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py"""
    # 从参数中提取提示语句
    prompt = params["prompt"]
    # 计算提示语句的长度
    l_prompt = len(prompt)
    # 替换提示语句中的特定前缀，使其适应模型输入格式
    prompt = prompt.replace("ai:", "assistant:").replace("human:", "user:")
    # 获取温度参数并转换为浮点数
    temperature = float(params.get("temperature", 1.0))
    # 获取生成的最大token数目
    max_new_tokens = int(params.get("max_new_tokens", 2048))
    # 获取停止条件字符串
    stop_str = params.get("stop", None)
    # 使用分词器对提示语句进行编码，获取初始的输入ids
    input_ids = tokenizer(prompt).input_ids
    # 复制输入ids到输出ids列表
    output_ids = list(input_ids)

    # 计算最大源文本长度
    max_src_len = context_len - max_new_tokens - 8
    # 裁剪输入ids以适应最大源文本长度
    input_ids = input_ids[-max_src_len:]

    # 循环生成新token
    for i in range(max_new_tokens):
        # 第一次迭代使用模型生成输出
        if i == 0:
            # 使用模型生成输出，保留缓存
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            # 获取输出的logits
            logits = out.logits
            # 获取缓存的键值对
            past_key_values = out.past_key_values
        else:
            # 创建用于attention的mask
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device
            )
            # 使用模型生成输出，保留缓存和注意力mask
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            # 获取输出的logits
            logits = out.logits
            # 获取缓存的键值对
            past_key_values = out.past_key_values

        # 获取最后一个token的logits
        last_token_logits = logits[0][-1]

        # 如果设备是"mps"，切换到CPU以避免mps后端的某些bug
        if device == "mps":
            last_token_logits = last_token_logits.float().to("cpu")

        # 根据温度参数选择token
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        # 将生成的token添加到输出ids列表中
        output_ids.append(token)

        # 如果生成的token是终止token，设置停止标志为True
        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        # 如果达到流输出间隔或者已生成所有token或者已停止，生成最终输出并yield返回
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            # 在最终输出中找到停止条件字符串的位置并截断
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        # 如果已停止，跳出循环
        if stopped:
            break

    # 删除缓存的键值对
    del past_key_values


@torch.inference_mode()
# 定义生成输出的函数，基于指定的模型、分词器、参数和设备
def generate_output(
    model, tokenizer, params, device, context_len=4096, stream_interval=2
):
    """Fork from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py"""

    # 从参数中提取提示语句
    prompt = params["prompt"]
    # 计算提示语句的长度
    l_prompt = len(prompt)
    # 获取温度参数并转换为浮点数
    temperature = float(params.get("temperature", 1.0))
    # 获取生成的最大token数目
    max_new_tokens = int(params.get("max_new_tokens", 2048))
    # 获取停止条件字符串
    stop_str = params.get("stop", None)

    # 使用分词器对提示语句进行编码，获取初始的输入ids
    input_ids = tokenizer(prompt).input_ids
    # 复制输入ids到输出ids列表
    output_ids = list(input_ids)

    # 计算最大源文本长度
    max_src_len = context_len - max_new_tokens - 8
    # 限制输入的长度为 max_src_len，取后面的部分
    input_ids = input_ids[-max_src_len:]

    # 对于新生成的最大 token 数量进行循环
    for i in range(max_new_tokens):
        # 如果是第一次循环
        if i == 0:
            # 使用模型生成输出，保留缓存
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            # 创建 attention mask，处理之前生成的关键值
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device
            )
            # 使用模型生成输出，保留缓存和之前的关键值
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        # 获取最后一个 token 的 logits
        last_token_logits = logits[0][-1]

        # 如果设备是 "mps"，将 logits 转为 float 类型并切换到 CPU 处理，避免某些 mps 后端的 bug
        if device == "mps":
            last_token_logits = last_token_logits.float().to("cpu")

        # 根据 temperature 生成 token
        if temperature < 1e-4:
            # 如果温度接近于 0，选择 logits 最大的 token
            token = int(torch.argmax(last_token_logits))
        else:
            # 使用 softmax 根据温度生成 token
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        # 将生成的 token 添加到输出的 id 列表中
        output_ids.append(token)

        # 如果生成的 token 是结束 token，则标记生成结束
        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        # 如果当前循环次数能被 stream_interval 整除，或者达到最大生成 token 数量，或者已经停止生成，则进行处理
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            # 将输出的 token 序列解码为文本
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            # 在指定位置 l_prompt 前找到停止字符串 stop_str 的最后出现位置
            pos = output.rfind(stop_str, l_prompt)
            # 如果找到停止字符串，则截取它之前的部分作为最终输出
            if pos != -1:
                output = output[:pos]
                stopped = True
            # 返回最终的生成文本输出
            return output

        # 如果已经停止生成，则退出循环
        if stopped:
            break

    # 删除生成过程中使用的过去的关键值
    del past_key_values
# 使用 Torch 的推断模式装饰器，指示下面的函数在推断模式下运行
@torch.inference_mode()
def generate_output_ex(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    # 从参数中获取提示文本
    prompt = params["prompt"]
    # 获取温度参数，用于控制生成文本的多样性
    temperature = float(params.get("temperature", 1.0))
    # 获取最大新生成标记数目的参数
    max_new_tokens = int(params.get("max_new_tokens", 2048))
    # 获取停止条件参数
    stop_parameter = params.get("stop", None)

    # 如果停止条件是 EOS 标记，则将其设为 None
    if stop_parameter == tokenizer.eos_token:
        stop_parameter = None

    stop_strings = []
    # 如果停止条件是字符串，则加入到停止字符串列表中
    if isinstance(stop_parameter, str):
        stop_strings.append(stop_parameter)
    # 如果停止条件是列表，则直接赋值给停止字符串列表
    elif isinstance(stop_parameter, list):
        stop_strings = stop_parameter
    # 如果停止条件为 None，则不做任何操作
    elif stop_parameter is None:
        pass
    else:
        # 如果停止条件既不是字符串也不是列表，则引发类型错误异常
        raise TypeError("Stop parameter must be string or list of strings.")

    # 使用分词器处理提示文本，获取输入标记的 IDs
    input_ids = tokenizer(prompt).input_ids
    output_ids = []

    # 计算最大源文本长度，用于生成新文本
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]  # 仅保留最后的 max_src_len 个标记作为输入
    stop_word = None

    # 循环生成新标记，直到达到最大生成标记数或满足停止条件
    for i in range(max_new_tokens):
        if i == 0:
            # 如果是第一次迭代，则使用完整的输入标记生成输出
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            # 对于后续迭代，将生成的标记作为输入生成输出
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        # 获取最后一个标记的 logits
        last_token_logits = logits[0][-1]

        # 根据温度参数选择下一个生成标记
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        # 检查生成的标记是否为结束标记
        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        # 将生成的标记序列解码为文本输出
        output = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 检查生成的文本是否包含任一停止字符串
        for stop_str in stop_strings:
            pos = output.rfind(stop_str)
            if pos != -1:
                # 如果找到停止字符串，则截取生成的文本，并标记为停止
                output = output[:pos]
                stopped = True
                stop_word = stop_str
                break
            else:
                pass

        # 如果满足停止条件，则结束生成
        if stopped:
            break

    # 清理过程中使用的过去关键值
    del past_key_values

    # 如果找到停止字符串，则返回截取后的输出
    if pos != -1:
        return output[:pos]
    else:
        return output


# 使用 Torch 的推断模式装饰器，指示下面的函数在推断模式下运行
@torch.inference_mode()
def get_embeddings(model, tokenizer, prompt):
    # 使用分词器处理提示文本，获取输入标记的 IDs
    input_ids = tokenizer(prompt).input_ids
    # 检测是否有可用的 GPU 设备，如果有则使用 CUDA，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取输入嵌入层，并移动到合适的设备上
    input_embeddings = model.get_input_embeddings().to(device)

    # 计算输入标记的嵌入向量
    embeddings = input_embeddings(torch.LongTensor([input_ids]).to(device))
    # 计算输入标记的嵌入向量的均值，并将结果移回 CPU
    mean = torch.mean(embeddings[0], 0).cpu().detach()
    return mean.to(device)
```