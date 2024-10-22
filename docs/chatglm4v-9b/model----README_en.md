# GLM-4V-9B

**2024/08/12, The repository code has been updated and now requires `transformers>=4.44.0`. Please update your dependencies accordingly.**

GLM-4V-9B is an open source multimodal version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.
**GLM-4V-9B** has the ability to conduct multi-round conversations in Chinese and English at a high resolution of 1120 * 1120. In multimodal evaluations of comprehensive Chinese and English abilities, perceptual reasoning, text recognition, and chart understanding, GLM-4V-9B has shown superior performance over GPT-4-turbo-2024-04-09, Gemini
1.0 Pro, Qwen-VL-Max, and Claude 3 Opus.

### Multimodal 

GLM-4V-9B is a multimodal language model with visual understanding capabilities. The evaluation results of its related classic tasks are as follows:


|                         | **MMBench-EN-Test** | **MMBench-CN-Test** | **SEEDBench_IMG** | **MMStar** | **MMMU** | **MME** | **HallusionBench** | **AI2D** | **OCRBench** |
|-------------------------|---------------------|---------------------|-------------------|------------|----------|---------|--------------------|----------|--------------|
|                         | 英文综合                | 中文综合                | 综合能力              | 综合能力       | 学科综合     | 感知推理    | 幻觉性                | 图表理解     | 文字识别         |
| **GPT-4o, 20240513**    | 83.4                | 82.1                | 77.1              | 63.9       | 69.2     | 2310.3  | 55                 | 84.6     | 736          |
| **GPT-4v, 20240409**    | 81                  | 80.2                | 73                | 56         | 61.7     | 2070.2  | 43.9               | 78.6     | 656          |
| **GPT-4v, 20231106**    | 77                  | 74.4                | 72.3              | 49.7       | 53.8     | 1771.5  | 46.5               | 75.9     | 516          |
| **InternVL-Chat-V1.5**  | 82.3                | 80.7                | 75.2              | 57.1       | 46.8     | 2189.6  | 47.4               | 80.6     | 720          |
| **LlaVA-Next-Yi-34B**   | 81.1                | 79                  | 75.7              | 51.6       | 48.8     | 2050.2  | 34.8               | 78.9     | 574          |
| **Step-1V**             | 80.7                | 79.9                | 70.3              | 50         | 49.9     | 2206.4  | 48.4               | 79.2     | 625          |
| **MiniCPM-Llama3-V2.5** | 77.6                | 73.8                | 72.3              | 51.8       | 45.8     | 2024.6  | 42.4               | 78.4     | 725          |
| **Qwen-VL-Max**         | 77.6                | 75.7                | 72.7              | 49.5       | 52       | 2281.7  | 41.2               | 75.7     | 684          |
| **GeminiProVision**     | 73.6                | 74.3                | 70.7              | 38.6       | 49       | 2148.9  | 45.7               | 72.9     | 680          |
| **Claude-3V Opus**      | 63.3                | 59.2                | 64                | 45.7       | 54.9     | 1586.8  | 37.8               | 70.6     | 694          |
| **GLM-4v-9B**           | 81.1                | 79.4                | 76.8              | 58.7       | 47.2     | 2163.8  | 46.6               | 81.1     | 786          |


**This repository is the model repository of GLM-4V-9B, supporting `8K` context length.**

## Quick Start

**For more inference code and requirements, please visit our [github page](https://github.com/THUDM/GLM-4).**

**Please strictly follow the [dependencies](https://github.com/THUDM/GLM-4/blob/main/basic_demo/requirements.txt) to install, otherwise it will not run properly**


```python

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)

query = 'discribe this image'
image = Image.open("your image").convert('RGB')
inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # chat mode

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4v-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
```

## License

The use of the GLM-4 model weights needs to comply with the [LICENSE](LICENSE).

## Citation

If you find our work helpful, please consider citing the following papers.

```
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, 
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```

```
@misc{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

