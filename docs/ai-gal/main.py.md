# `.\AI-GAL\main.py`

```py
import re  # 导入正则表达式模块
import base64  # 导入base64编解码模块
import os  # 导入操作系统相关功能的模块
import requests  # 导入发送HTTP请求的模块
import json  # 导入处理JSON数据的模块
import time  # 导入时间相关的模块
import configparser  # 导入配置文件解析模块
import random  # 导入生成随机数的模块

import renpy  # 导入Ren'Py引擎相关的模块

running_state = False  # 设置运行状态为假
background_list = []  # 初始化背景列表为空
if_already = False  # 设置条件变量if_already为假

character_list = []  # 初始化人物列表为空
game_directory = renpy.config.basedir  # 获取Ren'Py项目的基础目录
# game_directory = r"D:\renpy-8.1.1-sdk.7z\PROJECT"
game_directory = os.path.join(game_directory, "game")  # 拼接游戏目录路径
images_directory = os.path.join(game_directory, "images")  # 拼接图片目录路径
audio_directory = os.path.join(game_directory, "audio")  # 拼接音频目录路径
config = configparser.ConfigParser()  # 创建配置文件解析器对象
config.read(rf"{game_directory}\config.ini", encoding='utf-8')  # 读取配置文件

def gpt(system, prompt, mode="default"):
    config = configparser.ConfigParser()  # 创建配置文件解析器对象
    config.read(rf"{game_directory}\config.ini", encoding='utf-8')  # 读取配置文件
    key = config.get('CHATGPT', 'GPT_KEY')  # 获取CHATGPT部分的GPT_KEY配置项值
    url = config.get('CHATGPT', 'BASE_URL')  # 获取CHATGPT部分的BASE_URL配置项值
    model = config.get('CHATGPT', 'model')  # 获取CHATGPT部分的model配置项值

    payload = json.dumps({  # 构建JSON格式的数据payload
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    headers = {  # 定义HTTP请求的头部信息
        'Accept': 'application/json',
        'Authorization': f'Bearer {key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)  # 发送POST请求
    content = json.loads(response.text)['choices'][0]['message']['content']  # 解析响应JSON数据
    return content  # 返回处理后的内容

def separate_content(text):
    title_pattern = re.compile(r"标题:(.+)")  # 编译标题的正则表达式模式
    outline_pattern = re.compile(r"大纲:(.+?)背景:", re.DOTALL)  # 编译大纲的正则表达式模式
    background_pattern = re.compile(r"背景:(.+?)人物:", re.DOTALL)  # 编译背景的正则表达式模式
    characters_pattern = re.compile(r"人物:(.+)", re.DOTALL)  # 编译人物的正则表达式模式
    title = title_pattern.search(text).group(1).strip()  # 匹配并提取标题内容
    outline = outline_pattern.search(text).group(1).strip()  # 匹配并提取大纲内容
    background = background_pattern.search(text).group(1).strip()  # 匹配并提取背景内容
    characters = characters_pattern.search(text).group(1).strip()  # 匹配并提取人物内容
    information = (title, outline, background, characters)  # 将信息打包为元组
    return information  # 返回解析后的信息

# ----------------------------------------------------------------------
online_draw_key = config.get('AI绘画', '绘画key')  # 获取AI绘画部分的绘画key配置项值

url = "https://cn.tensorart.net/v1/jobs"  # 定义请求的URL地址
headers = {  # 定义HTTP请求的头部信息
    "Content-Type": "application/json; charset=UTF-8",
    "Authorization": f"Bearer 0d2977d8d84048f5a8102fdd5c7ddd1d"
}

def online_generate(prompt, mode):
    print("云端启动绘画")  # 输出提示信息
    # TMND:611399039965066695
    # 天空:611437926598989702
    requests_id = ''.join([str(random.randint(0, 9)) for _ in range(10)])  # 生成随机请求ID
    if mode == 'background':  # 如果模式为背景模式
        width = 960  # 设置宽度为960像素
        height = 540  # 设置高度为540像素
        prompt2 = "(no_human)" + prompt  # 构建新的提示内容
    #     if config.get('AI绘画', '人物绘画模型ID(本地模式不填)'):
    #         model = config.get('AI绘画', '人物绘画模型ID(本地模式不填)')
    #     else:
    #         model = "611399039965066695"
    #
    else:
        # 设置默认的图像宽度和高度
        width = 512
        height = 768
        # 修改提示文字以包含"(upper_body),solo"前缀
        prompt2 = "(upper_body),solo" + prompt
        # 根据配置获取AI绘画的背景绘画模型ID，如果未设置则使用默认模型ID
        # if config.get('AI绘画', '背景绘画模型ID(本地模式不填)'):
        #     model = config.get('AI绘画', '背景绘画模型ID(本地模式不填)')
        # else:
        #     model = "700862942452666384"

    # 准备发送的数据结构
    data = {
        "request_id": str(requests_id),  # 转换请求ID为字符串形式
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "seed": -1,  # 初始化随机种子为-1
                    "count": 1    # 设置计数为1
                }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": width,             # 设置扩散宽度
                    "height": height,           # 设置扩散高度
                    "prompts": [{"text": prompt2}],  # 使用修改后的提示文字
                    "steps": 25,                # 执行步数为25步
                    "sdVae": "animevae.pt",     # 设置sdVae参数
                    "sd_model": "611399039965066695",  # 设置sd_model参数
                    "clip_skip": 2,             # 设置clip_skip参数
                    "cfg_scale": 7              # 设置cfg_scale参数
                }
            },
            {
                "type": "IMAGE_TO_UPSCALER",
                "image_to_upscaler": {
                    "hr_upscaler": "R-ESRGAN 4x+ Anime6B",  # 高分辨率图像放大器名称
                    "hr_scale": 2,              # 高分辨率缩放比例
                    "hr_second_pass_steps": 10,  # 第二次处理的步骤数
                    "denoising_strength": 0.3   # 降噪强度设置
                }
            }
        ]
    }

    # 发送POST请求到指定的URL，并获取响应
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # 检查响应状态码
    if response.status_code == 200:
        # 从响应的JSON数据中提取作业ID并返回
        id = json.loads(response.text)['job']['id']
        return id
    else:
        # 打印请求失败的状态码
        print(f"请求失败，状态码：{response.status_code}，请检查是否正确填写了key")
def generate_image(prompt, image_name, mode):
    global images_directory  # 声明全局变量 images_directory

    url = "http://127.0.0.1:7860"  # 设定本地服务器的 URL

    if mode == 'background':  # 如果 mode 参数为 'background'
        width = 960  # 设置图像宽度为 960
        height = 540  # 设置图像高度为 540
        prompt2 = "(no_human)"  # 设置第二个提示为 "(no_human)"
        model = "tmndMix_tmndMixVPruned.safetensors [d9f11471a8]"  # 设置模型名称为指定的字符串
    else:
        # 如果条件不满足，则设置默认的宽度和高度
        width = 512
        height = 768
        # 设置默认的图像生成提示语句
        prompt2 = "(upper_body),solo"
        # 设置默认的模型名称
        model = "天空之境.safetensors [c1d961233a]"

    # 构建请求的 payload 参数
    payload = {
        # 图像生成的主要提示语句，包括简单背景和其他指定特征
        "prompt": f"masterpiece,wallpaper,simple background,{prompt},{prompt2}",
        # 负面提示语句，用于生成过程中的限制条件
        "negative_prompt": "Easynagative,bad,worse,nsfw",
        # 生成图像的步骤数
        "steps": 30,
        # 使用的采样器名称
        "sampler_name": "DPM++ 2M SDE",
        # 图像宽度
        "width": width,
        # 图像高度
        "height": height,
        # 是否还原面部特征
        "restore_faces": False,
        # 是否启用高分辨率生成
        "enable_hr": True,
        # 使用的高分辨率增强器名称
        "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
        # 高分辨率生成的缩放比例
        "hr_scale": 2,
        # 高分辨率生成的第二遍处理步数
        "hr_second_pass_steps": 15,
        # 去噪强度参数
        "denoising_strength": 0.3
    }

    try:
        # 发送 POST 请求，生成图像
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        # 如果请求成功
        if response.status_code == 200:
            # 解析响应的 JSON 数据
            r = response.json()
            # 遍历生成的图像数据
            for i, img_data in enumerate(r['images']):
                # 处理 base64 编码的图像数据
                if ',' in img_data:
                    base64_data = img_data.split(",", 1)[1]
                else:
                    base64_data = img_data
                # 解码 base64 数据为图像二进制
                image_data = base64.b64decode(base64_data)
                # 最终保存的图像文件名
                final_image_name = f'{image_name}.png'
                # 将图像数据写入文件
                with open(fr'{images_directory}\{final_image_name}', 'wb') as f:
                    f.write(image_data)
                # 打印保存成功信息
                print(f'图片已保存为 {final_image_name}')

        else:
            # 若生成图像请求失败，打印错误信息
            print("Failed to generate image:", response.text)

    except:
        # 捕获所有异常情况，打印绘图失败信息
        print("绘图失败！")
# 读取音频生成配置文件中的路径信息
def generate_audio(response, name, output_name):
    # 声明全局变量，指定音频文件输出目录
    global audio_directory
    # 读取和解析配置文件
    config = configparser.ConfigParser()
    config.read(rf"{game_directory}\config.ini", encoding='utf-8')

    # 准备 JSON 数据，包含 GPT 和 SOVITS 模型路径
    json_data = {
        "gpt_model_path": config.get('SOVITS', 'gpt_model_path'),
        "sovits_model_path": config.get('SOVITS', 'sovits_model_path')
    }

    # 根据名称选择正确的 URL
    if name == 1:
        url = config.get('SOVITS', 'sovits_url1').format(response=response)
    elif name == 2:
        url = config.get('SOVITS', 'sovits_url2').format(response=response)
    elif name == 3:
        url = config.get('SOVITS', 'sovits_url3').format(response=response)
    elif name == 4:
        url = config.get('SOVITS', 'sovits_url4').format(response=response)
    elif name == 5:
        url = config.get('SOVITS', 'sovits_url5').format(response=response)
    else:
        url = config.get('SOVITS', 'sovits_url6').format(response=response)

    # 发送 POST 请求，将 JSON 数据发送到本地服务器
    requests.post('http://127.0.0.1:9880/set_model', json=json_data)

    try:
        # 发送 GET 请求获取音频数据，将其写入到指定的输出文件中
        response = requests.get(url)
        with open(rf'{audio_directory}\{output_name}.wav', 'wb') as file:
            file.write(response.content)

    except Exception as e:
        # 捕获并打印任何异常
        print("语音错误", e)


# 将对话信息添加到 JSON 文件中
def add_dialogue_to_json(character, text, background_image, audio):
    # 声明全局变量，指定游戏目录
    global game_directory

    try:
        # 打开并读取现有的对话 JSON 文件
        with open(rf"{game_directory}\dialogues.json", "r", encoding="utf-8") as file:
            dialogues = json.load(file)

        # 将新的对话信息添加到对话列表中
        dialogues["conversations"].append({
            "character": character,
            "text": text,
            "background_image": background_image,
            "audio": audio
        })

        # 将更新后的对话 JSON 数据写回文件
        with open(rf"{game_directory}\dialogues.json", "w", encoding="utf-8") as file:
            json.dump(dialogues, file, indent=4, ensure_ascii=False)

        print("新的对话已成功添加到dialogues.json文件中")

    except FileNotFoundError:
        # 捕获并打印文件未找到异常
        print("错误:找不到文件 dialogues.json")

    except Exception as e:
        # 捕获并打印其他异常
        print(f"发生错误:{e}")


# 使用服务移除图像背景
def rembg(pic):
    # 声明全局变量，指定图像文件目录
    global images_directory
    # 指定用于发送 POST 请求的 URL
    url = "http://localhost:7000/api/remove"
    # 构造完整的图像文件路径
    file_path = rf"{images_directory}/{pic}.png"

    # 使用文件对象发送 POST 请求并接收响应
    with open(file_path, 'rb') as file:
        response = requests.post(url, files={'file': file})

    # 将接收到的响应内容写入原始图像文件
    with open(file_path, 'wb') as output_file:
        output_file.write(response.content)


# 选择故事并生成选项文本文件
def choose_story():
    # 打开并读取故事文本文件
    with open(rf"{game_directory}\story.txt", 'r', encoding='utf-8') as file:
        book = file.read()

    # 使用模型生成故事选项，并格式化返回的选项文本
    choices = gpt("你是galgame剧情家，精通各种galgame写作",
                  f"根据galgame剧情,以男主角的视角，设计男主角接下来的三个分支选项。内容:{book},返回格式:1.xxx\n2.xxx\n3.xxx,要求每个选项尽量简短。不要使用markdown语法。")
    cleaned_text = '\n'.join([line.split('. ', 1)[1] if '. ' in line else line for line in choices.strip().split('\n')])

    # 将生成的选项文本写入文件
    with open(rf"{game_directory}\choice.txt", 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

    # 返回清理后的选项文本
    return cleaned_text


# 主函数入口
def main():
    # 声明全局变量，指定书籍目录、游戏目录、是否已存在和角色列表
    global book, game_directory, if_already, character_list
    # 读取和解析配置文件
    config = configparser.ConfigParser()
    # 使用给定的 game_directory 变量拼接路径，读取并解析指定的 config.ini 文件
    config.read(rf"{game_directory}\config.ini", encoding='utf-8')
    
    # 使用指定路径创建一个新的 dialogues.json 文件，并写入初始空的 JSON 结构
    with open(rf'{game_directory}\dialogues.json', 'w') as file:
        file.write("""{\n"conversations": [\n]\n}""")
    
    # 使用指定路径创建一个新的 characters.txt 文件，并清空文件内容
    with open(rf"{game_directory}\characters.txt", 'w') as file:
        file.write('')
    
    # 从 config 对象中获取 '剧情' 部分的 '剧本的主题' 设置，并赋值给 theme 变量
    theme = config.get('剧情', '剧本的主题')
    
    # 使用 gpt 函数生成标题、大纲、背景和人物信息，返回的是一个包含这些内容的元组
    title, outline, background, characters = separate_content(
        gpt("现在你是一名gal game剧情设计师，精通写各种各样的gal game剧情，不要使用markdown格式",
            f"现在请你写一份gal game的标题，大纲，背景，人物,我给出的主题和要求是{theme}，你的输出格式为:标题:xx\n大纲:xx\n背景:xx\n人物:xx(每个人物占一行,人物不多于5人)，每个人物的格式是人物名:介绍,无需序号。男主角也要名字").replace(
            "：", ":"))
    
    # 使用 gpt 函数生成第一章的文本内容，包括标题、大纲、背景和人物信息，以对话模式输出
    book = gpt("现在你是一名galgame剧情作家，精通写各种各样的galgame剧情，请不要使用markdown格式",
               f"现在根据以下内容开始写第一章:gal game标题:{title},gal game大纲:{outline},gal game背景:{background},galgame角色:{characters}。你的输出格式应该为对话模式，例xxx:xx表达，你的叙事节奏要非常非常慢，可以加一点新的内容进去。需要切换地点时，在句尾写[地点名]，[地点名]不可单独一行，不用切换则不写，开头第一句是旁白而且必须要包含地点[]，地点理论上不应该超过3处。不需要标题。输出例子:旁白:xxx[地点A]\n角色A:xxx\n角色B:xxx\n角色:xxx[地点B]\n旁白:xxx，角色名字要完整。")
    
    # 将生成的文本按行分割，并去掉最后一行
    lines = book.split('\n')
    book = '\n'.join(lines[:-1])
    
    # 将 book 按行分割为列表
    booklines = book.splitlines()
    
    # 打印 book 变量的内容
    print(book)
    
    # 将生成的文本写入到指定路径的 story.txt 文件中，使用 UTF-8 编码
    with open(rf"{game_directory}\story.txt", 'w', encoding='utf-8') as file:
        file.write(f"{book}\n")
    
    # 将角色信息写入到指定路径的 character_info.txt 文件中，使用 UTF-8 编码
    with open(rf"{game_directory}\character_info.txt", 'w', encoding='utf-8') as file:
        file.write(characters)
    
    # 将角色信息按行分割为列表
    characterslines = characters.splitlines()
    
    # 过滤掉 characterslines 列表中不包含冒号的行
    characterslines = [item for item in characterslines if ":" in item]
    
    # 打印过滤后的 characterslines 列表
    print(characterslines)
    
    # 遍历 characterslines 列表的索引范围
    for i in range(len(characterslines)):
        # 使用 gpt 函数生成相应人物形象的描述
        prompt = gpt(
            "根据人物设定给出相应的人物形象，应该由简短的英文单词或短句组成，输出格式样例:a girl,pink hair,black shoes,long hair,young,lovely。请注意，人名与实际内容无关无需翻译出来，只输出英文单词，不要输出多余的内容",
            f"人物形象{characterslines[i]}")
    
        # 从 characterslines[i] 中获取人物名字，去除非中文字符，标准化名字
        name = characterslines[i].split(":", 1)[0]
        name = re.sub(r'[^\u4e00-\u9fa5]', '', name)
    
        # 根据配置中的云端模式选择生成图片的方法
        if config.getboolean('AI绘画', '云端模式'):
            generate_image_pro(prompt, name, "character")
        else:
            generate_image(prompt, name, "character")
    
        # 调用 rembg 函数处理生成的图片
        rembg(name)
    
        # 将处理后的人物名字添加到 character_list 列表中
        character_list.append(name)
    
        # 将人物名字写入到 characters.txt 文件中
        with open(rf"{game_directory}\characters.txt", "a", encoding='utf-8') as file:
            file.write(f"{name}\n")
    # 遍历书籍文本的每一行
    for i in booklines:
        # 如果当前行不为空白行
        if i.strip() != "":
            # 提取方括号中的内容作为背景信息
            background = re.findall(r'(?<=\[).*?(?=\])', i)

            # 如果存在背景信息并且该背景未被记录过
            if background and background[0] not in background_list:
                # 使用 GPT 模型生成关于背景信息的翻译提示
                prompt = gpt(
                    "把下面的内容翻译成英文并且变成短词,比如red,apple,big这样。请注意，地名与实际内容无关无需翻译出来，例如星之学院应该翻译成academy而不是star academy。下面是你要翻译的内容:",
                    background[0])
                # 打印生成的翻译提示
                print(prompt)
                # 根据配置选择生成背景图像的方式（云端或本地）
                if config.getboolean('AI绘画', '云端模式'):
                    generate_image_pro(prompt, background[0], "background")
                else:
                    generate_image(prompt, background[0], "background")
                # 记录当前处理的背景图像
                background_image = background[0]
                # 将处理过的背景信息添加到列表中
                background_list.append(background_image)

            else:
                # 如果背景信息已经记录过或者不存在，则将背景图像置为空字符串
                background_image = ""

            # 将中文冒号替换为英文冒号
            i = i.replace("：", ":")

            # 如果当前行中不包含英文冒号，则在行首添加旁白标识
            i = "旁白:" + i if ":" not in i else i

            # 根据冒号分割行为人物和对话内容
            character, text = i.split(":", 1)
            # 删除方括号及其中内容，并保存为text1
            text1 = re.sub(r'\[.*?\]', '', text)
            # 删除圆括号及其中内容，并保存为text2，同时处理圆括号的全角半角
            text2 = re.sub(r'\（[^)]*\）', '', text1.replace("(", "（").replace(")", "）"))

            try:
                # 在人物列表中查找人物名，确定其在音频文件中的编号
                index = character_list.index(character)
                audio_num = index + 1

            except ValueError:
                # 如果人物不在列表中，打印错误信息，并将音频编号设为6
                print(f"{character} 不在列表中")
                audio_num = 6

            # 如果当前行的角色不是旁白
            if character != "旁白":
                # 根据配置选择生成语音的方式（云端或本地）
                if config.getboolean('SOVITS', '云端模式'):
                    generate_audio_pro(text2, audio_num, text1)
                else:
                    generate_audio(text2, audio_num, text1)

            # 如果角色是旁白，则将角色名设为空字符串
            character = "" if character == "旁白" else character

            # 如果对话内容不为空，则将对话信息添加到 JSON 文件中
            if text != "":
                add_dialogue_to_json(character, text2, background_image, text1)

    # 选择故事（函数调用）
    choose_story()
    # 设置一个标志，表示已经处理完毕
    if_already = True
# 定义一个函数用于继续故事的进行，根据选择的分支进行处理
def story_continue(choice):
    # 声明全局变量
    global book, running_state, game_directory, character_list
    running_state = True  # 设置运行状态为True，表示程序正在运行

    # 打开并读取故事文本文件
    with open(rf"{game_directory}\story.txt", 'r', encoding='utf-8') as file:
        book = file.read()

    # 打开并读取角色信息文本文件
    with open(rf"{game_directory}\character_info.txt", 'r', encoding='utf-8') as file:
        character_info = file.read()

    # 使用GPT模型生成新的剧情文本
    add_book = gpt(
        "现在你是一名galgame剧情设计师，精通写各种各样的galgame剧情。只输出文本，不要输出任何多余的。不要使用markdown格式，如果需要切换场景在对话的后面加上[地点]，输出例子:旁白:xxx[地点A]\n角色A:xxx\n角色B:xxx\n角色:xxx[地点B]\n旁白:xxx，角色名字要完整。",
        f"请你根据以下内容继续续写galgame剧情。只返回剧情。人物设定：{character_info}，内容:{book},我选则的分支是{choice}")

    # 将生成的剧情文本按行分割并更新到整体剧本中
    booklines = add_book.splitlines()
    book = book + "\n" + add_book

    # 将更新后的剧本写回story.txt文件
    with open(rf'{game_directory}\story.txt', 'w', encoding='utf-8') as file:
        file.write(f"{book}\n")

    # 遍历剧情文本的每一行
    for i in booklines:
        if i.strip() != "":  # 如果行内容非空
            # 从文本中提取方括号中的地点信息
            background = re.findall(r'(?<=\[).*?(?=\])', i)

            # 如果存在地点信息且地点不在列表中，生成对应背景图像
            if background and background[0] not in background_list:
                # 使用GPT模型翻译地点信息为短词
                prompt = gpt(
                    "把下面的内容翻译成英文并且变成短词,比如red,apple,big这样。请注意，地名与实际内容无关无需翻译出来，例如星之学院应该翻译成academy而不是star academy。下面是你要翻译的内容:",
                    background[0])
                print(prompt)

                # 根据配置选择生成图像的方式
                cloud_mode = config.getboolean('AI绘画', '云端模式')
                if cloud_mode:
                    generate_image_pro(prompt, background[0], "background")
                else:
                    generate_image(prompt, background[0], "background")

                # 将生成的背景图像信息添加到列表中
                background_image = background[0]
                background_list.append(background_image)

            else:
                background_image = ""

            # 将文本中的全角冒号替换为半角冒号
            i = i.replace("：", ":")

            # 如果行中不包含冒号，添加"旁白:"标识
            i = "旁白:" + i if ":" not in i else i

            # 按冒号将行分割为角色和对话文本
            character, text = i.split(":", 1)

            # 去除对话文本中方括号中的地点信息
            text1 = re.sub(r'\[.*?\]', '', text)

            # 去除对话文本中圆括号中的内容
            text2 = re.sub(r'\（[^)]*\）', '', text1.replace("(", "（").replace(")", "）"))

            # 尝试打开角色信息文件，查找角色名字并记录行号作为音频编号
            try:
                with open(rf"{game_directory}\characters.txt", 'r', encoding='utf-8') as file:
                    line_number = 0
                    for line in file:
                        line_number += 1
                        if character in line:
                            audio_num = line_number
            except ValueError:
                # 如果角色不在列表中，打印错误信息并将音频编号设置为默认值6
                print(f"{character} 不在列表中")
                audio_num = 6

            # 如果角色不是"旁白"或"new"，根据配置生成对应的音频文件
            if character != "旁白" and character != "new":
                if config.getboolean('SOVITS', '云端模式'):
                    generate_audio_pro(text2, audio_num, text1)
                else:
                    generate_audio(text2, audio_num, text1)

            # 如果对话文本非空，将对话信息添加到JSON文件中
            if text != "":
                add_dialogue_to_json(character, text2, background_image, text1)

    # 选择下一个剧情分支
    choose_story()

    # 设置运行状态为False，表示程序运行结束
    running_state = False
```