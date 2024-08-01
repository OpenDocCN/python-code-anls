# `.\DB-GPT-src\dbgpt\model\llm_out\gorilla_llm.py`

```py
import torch  # 导入PyTorch库


@torch.inference_mode()  # 设置函数为推理模式，这在使用PyTorch模型进行推断时很重要
def generate_stream(
    model, tokenizer, params, device, context_len=42048, stream_interval=2
):
    """Fork from https://github.com/ShishirPatil/gorilla/blob/main/inference/serve/gorilla_cli.py"""
    # 从给定的GitHub仓库中派生的代码，用于流式生成文本的CLI

    prompt = params["prompt"]  # 获取参数字典中的提示语句
    l_prompt = len(prompt)  # 计算提示语句的长度
    max_new_tokens = int(params.get("max_new_tokens", 1024))  # 获取生成的最大新token数量，默认为1024
    stop_str = params.get("stop", None)  # 获取停止生成的条件字符串，如果没有指定则为None

    input_ids = tokenizer(prompt).input_ids  # 使用分词器对提示语句进行编码得到输入的token IDs
    output_ids = list(input_ids)  # 将输入的token IDs转换为输出的token IDs的初始列表
    input_echo_len = len(input_ids)  # 记录输入token IDs的长度
    max_src_len = context_len - max_new_tokens - 8  # 计算最大的源文本长度

    input_ids = input_ids[-max_src_len:]  # 截取输入token IDs，以适应最大的源文本长度
    past_key_values = out = None  # 初始化过去键值和输出

    for i in range(max_new_tokens):  # 迭代生成新的token
        if i == 0:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)  # 对输入token IDs进行模型推理，返回输出
            logits = out.logits  # 获取输出的logits
            past_key_values = out.past_key_values  # 获取输出的过去键值
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),  # 将当前token转换为张量，传递给模型进行推理
                use_cache=True,
                past_key_values=past_key_values,  # 使用过去的键值进行推理
            )
            logits = out.logits  # 获取输出的logits
            past_key_values = out.past_key_values  # 获取输出的过去键值

        last_token_logits = logits[0][-1]  # 获取最后一个token的logits

        probs = torch.softmax(last_token_logits, dim=-1)  # 对最后一个token的logits进行softmax得到概率分布
        token = int(torch.multinomial(probs, num_samples=1))  # 根据概率分布进行多项式抽样，得到下一个token
        output_ids.append(token)  # 将生成的token添加到输出token列表中

        if token == tokenizer.eos_token_id:  # 如果生成的token是结束标记
            stopped = True  # 设置停止标志为True
        else:
            stopped = False  # 否则停止标志为False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            tmp_output_ids = output_ids[input_echo_len:]  # 获取除去输入token后的输出token列表
            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )  # 使用分词器对输出token列表进行解码得到生成文本
            pos = output.rfind(stop_str, l_prompt)  # 在生成文本中查找停止字符串最后出现的位置
            if pos != -1:
                output = output[:pos]  # 如果找到停止字符串，则截取生成文本
                stopped = True  # 设置停止标志为True
            yield output  # 返回生成的文本

        if stopped:
            break  # 如果已停止生成，则退出循环

    del past_key_values  # 删除过去键值，释放内存
```