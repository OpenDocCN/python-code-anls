<!--yml
category: 未分类
date: 2023-10-10 23:17:51
-->

# ChatGLM2 源码分析：`ChatGLMForConditionalGeneration.chat, .stream_chat`_绝不原创的飞龙的博客-CSDN博客

> 来源：[https://blog.csdn.net/wizardforcel/article/details/132841359](https://blog.csdn.net/wizardforcel/article/details/132841359)

### `.chat`

调用分析：

```
In [1]: q = '你好'

In [2]: r, his = model.chat(tokenizer, q)

In [3]: r
Out[3]: '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'

In [4]: his
Out[4]: [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。')]

In [5]: q = '你可以做什么？'

In [6]: r, his = model.chat(tokenizer, q, his)

In [7]: r
Out[7]: '我是一个大型语言模型，可以进行自然语言处理和生成。具体来说，我可以：\n\n1\.  回答问题：像人类一样回答您的问题，或者提供 相关信息。\n\n2\.  提供建议：根据您的问题提供一些建议，或者提供一些参考信息。\n\n3\.  进行翻译：将一种语言翻译成另一种语言，或者将一种语言的文本翻译成另一种语言的文本。\n\n4\.  生成文本：根据您的问题生成一些文本，比如文章、故事、新闻报道等。\n\n5\.  自动文本摘要：自动概括文本的内容，并生成摘要。\n\n6\.  情感分析：判断文本中情感的程度，并返回相应的情感信息。\n\n7\.  智能对话：进行智能对话，与人类交流并完成任务。\n\n请注意，我是一个机器，我的回答可能不够准确，也可能会有所误导。'

In [8]: his
Out[8]:
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'),
 ('你可以做什么？',
  '我是一个大型语言模型，可以进行自然语言处理和生成。具体来说，我可以：\n\n1\.  回答问题：像人类一样回答您的问题，或者提供相关信息 。\n\n2\.  提供建议：根据您的问题提供一些建议，或者提供一些参考信息。\n\n3\.  进行翻译：将一种语言翻译成另一种语言，或者将一种语言的文本翻译成另一种语言的文本。\n\n4\.  生成文本：根据您的问题生成一些文本，比如文章、故事、新闻报道等。\n\n5\.  自动文本摘要：自动概括文本的内容，并生成摘要。\n\n6\.  情感分析：判断文本中情感的程度，并返回相应的情感信息。\n\n7\.  智能对话：进行智能对话，与人类交流并完成任务。\n\n请注意，我是一个机器，我的回答可能不够准确，也可能会有所误导。')] 
```py

源码：

```
 @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())

        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        inputs = self.build_inputs(tokenizer, query, history=history)

        outputs = self.generate(**inputs, **gen_kwargs)

        '''
        prompt: '你好, output: tensor([[64790, 64792,   790, 30951,   517, 30910, 30939, 30996,    13,    13,
         54761, 31211, 39701,    13,    13, 55437, 31211, 36474, 54591,   243,
           162,   148,   142, 31404, 33030, 34797, 42481, 22011, 10461, 30944,
         30943, 30941, 30978, 30949, 31123, 48895, 35214, 54622, 31123, 32616,
         39905, 31901, 31639, 31155,     2]], device='cuda:0')
        tokenizer.decode(output[0]): '[Round 1]\n\n问：你好\n\n答： 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'
        '''
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]

        response = tokenizer.decode(outputs)

        response = self.process_response(response)

        history = history + [(query, response)]
        return response, history

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        '''
        将历史问答和当前提问组装成整个输入
        In [1]: tokenizer.build_prompt('Q3', [('Q1', 'A1'),('Q2', 'A2')])
        Out[1]: '[Round 1]\n\n问：Q1\n\n答：A1\n\n[Round 2]\n\n问：Q2\n\n答：A2\n\n[Round 3]\n\n问：Q3\n\n答：'
        '''
        prompt = tokenizer.build_prompt(query, history=history)
        '''
        整个提问传给分词器得到单词ID
        In [2]: tokenizer(['你好'], return_tensors="pt")
        Out[2]: {
           'input_ids': tensor([[64790, 64792, 36474, 54591]]), 
           'attention_mask': tensor([[1, 1, 1, 1]]), 
           'position_ids': tensor([[0, 1, 2, 3]])
        }
        '''
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs 
```py

### `.stream_chat`

调用分析：

```
In [133]: q = '你好'

In [134]: it = model.stream_chat(tokenizer, q)

In [135]: for r, his in it: print(r); print(his)
你
[('你好', '你')]
你好
[('你好', '你好')]
你好👋
[('你好', '你好👋')]
...
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题')]
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。')]
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。')]

In [136]: q = '你可以做什么？'

In [137]: it = model.stream_chat(tokenizer, q, his)

In [138]: for r, his in it: print(r); print(his)
我
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我')]
我是一款
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款')]
我是一款大型
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型')]
...
我是一款大型语言模型，可以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型语言模型，可 以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通')]
我是一款大型语言模型，可以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型语言模型，可 以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。')]
我是一款大型语言模型，可以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。
[('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型语言模型，可 以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。')] 
```py

源码：

```
 @torch.inference_mode()
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values=None,
                    max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
                    return_past_key_values=False, **kwargs):

        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if past_key_values is None and not return_past_key_values:

            inputs = self.build_inputs(tokenizer, query, history=history)
        else:

            inputs = self.build_stream_inputs(tokenizer, query, history=history)
        if past_key_values is not None:

            past_length = past_key_values[0][0].shape[0]

            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len

            inputs.position_ids += past_length

            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs['attention_mask'] = attention_mask
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                            return_past_key_values=return_past_key_values, **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs

            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            '''
            q: '你好'
            iter1 response: '你'
            iter2 response: '你好'
            ...
            iterN response: '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'
            '''
            response = tokenizer.decode(outputs)

            if response and response[-1] != "�":

                response = self.process_response(response)

                new_history = history + [(query, response)]
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):

        if history:

            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            '''
            将 prompt 转成单词 ID，去掉开头的 ID64790、ID64792
            In [147]: tokenizer.encode('\n\n你好', add_special_tokens=False)
            Out[147]: [30910, 13, 13, 39701]
            In [149]: tokenizer.encode('\n\n你好')
            Out[149]: [64790, 64792, 30910, 13, 13, 39701]
            '''
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)

            input_ids = input_ids[1:]
            '''
            为 input_ids 生成相应的 attention_mask 和 position_ids
            In [151]: tokenizer.batch_encode_plus(
                [([13,13,39701], None)], 
                return_tensors="pt", 
                add_special_tokens=False
            )
            Out[151]: {
                'input_ids': tensor([[   13,    13, 39701]]), 
                'attention_mask': tensor([[1, 1, 1]]), 
                'position_ids': tensor([[0, 1, 2]])
            }
            '''
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
        else:

            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs 
```