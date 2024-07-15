# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\gradioDemo\app.py`

```py
# 导入 zipfile 模块，用于处理 ZIP 文件
import zipfile
# 导入 gradio 库，用于构建交互式界面
import gradio as gr
# 导入 PIL 库中的 Image 类，用于处理图像文件
from PIL import Image
# 导入自定义模块 chatharuhi 中的 ChatHaruhi 类
from chatharuhi import ChatHaruhi
# 导入 wget 库，用于从 URL 下载文件
import wget
# 导入 os 模块，用于操作系统相关的功能
import os
# 导入 openai 模块，尚未使用，可能用于其他功能
import openai
# 导入 copy 模块，用于深拷贝对象
import copy

# 定义角色名称到英文名的映射字典
NAME_DICT = {'汤师爷': 'tangshiye', '慕容复': 'murongfu', '李云龙': 'liyunlong', 'Luna': 'Luna', '王多鱼': 'wangduoyu',
             'Ron': 'Ron', '鸠摩智': 'jiumozhi', 'Snape': 'Snape',
             '凉宫春日': 'haruhi', 'Malfoy': 'Malfoy', '虚竹': 'xuzhu', '萧峰': 'xiaofeng', '段誉': 'duanyu',
             'Hermione': 'Hermione', 'Dumbledore': 'Dumbledore', '王语嫣': 'wangyuyan',
             'Harry': 'Harry', 'McGonagall': 'McGonagall', '白展堂': 'baizhantang', '佟湘玉': 'tongxiangyu',
             '郭芙蓉': 'guofurong', '旅行者': 'wanderer', '钟离': 'zhongli',
             '胡桃': 'hutao', 'Sheldon': 'Sheldon', 'Raj': 'Raj', 'Penny': 'Penny', '韦小宝': 'weixiaobao',
             '乔峰': 'qiaofeng', '神里绫华': 'ayaka', '雷电将军': 'raidenShogun', '于谦': 'yuqian'}

# 尝试创建 "characters_zip" 文件夹，如果已存在则跳过
try:
    os.makedirs("characters_zip")
except:
    pass

# 尝试创建 "characters" 文件夹，如果已存在则跳过
try:
    os.makedirs("characters")
except:
    pass

# 创建一个空字典，用于存储每个角色的 ChatHaruhi 对象
ai_roles_obj = {}

# 遍历 NAME_DICT 字典中的每个角色英文名
for ai_role_en in NAME_DICT.values():
    # 构造角色数据的下载 URL
    file_url = f"https://github.com/LC1332/Haruhi-2-Dev/raw/main/data/character_in_zip/{ai_role_en}.zip"
    
    # 尝试创建 "characters/<角色英文名>" 文件夹，如果已存在则跳过
    try:
        os.makedirs(f"characters/{ai_role_en}")
    except:
        pass
    
    # 如果角色对应的 ZIP 文件不在 "characters_zip" 文件夹中
    if f"{ai_role_en}.zip" not in os.listdir(f"characters_zip"):
        # 下载 ZIP 文件到指定目标文件
        destination_file = f"characters_zip/{ai_role_en}.zip"
        wget.download(file_url, destination_file)
        
        # 解压 ZIP 文件到指定目标文件夹
        destination_folder = f"characters/{ai_role_en}"
        with zipfile.ZipFile(destination_file, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
    
    # 设置角色数据的存储文件夹路径和系统提示文件路径
    db_folder = f"./characters/{ai_role_en}/content/{ai_role_en}"
    system_prompt = f"./characters/{ai_role_en}/content/system_prompt.txt"
    
    # 创建 ChatHaruhi 对象并加入到 ai_roles_obj 字典中
    ai_roles_obj[ai_role_en] = ChatHaruhi(system_prompt=system_prompt,
                                          llm="openai",
                                          story_db=db_folder,
                                          verbose=True)

# 异步函数：根据用户角色、用户文本、AI 角色和当前聊天历史获取 AI 的响应
async def get_response(user_role, user_text, ai_role, chatbot):
    # 获取 AI 角色对应的英文名
    role_en = NAME_DICT[ai_role]
    # 深拷贝当前聊天历史到 AI 对象中
    ai_roles_obj[role_en].dialogue_history = copy.deepcopy(chatbot)
    # 使用 AI 对象生成响应
    response = ai_roles_obj[role_en].chat(role=user_role, text=user_text)
    # 构造用户消息字符串，并将用户消息和 AI 响应加入到聊天历史中
    user_msg = user_role + ':「' + user_text + '」'
    chatbot.append((user_msg, response))
    # 返回更新后的聊天历史
    return chatbot

# 异步函数：根据用户角色、用户文本、AI 角色和当前聊天历史获取 AI 的响应，并返回响应和空值
async def respond(user_role, user_text, ai_role, chatbot):
    return await get_response(user_role, user_text, ai_role, chatbot), None

# 函数：清空当前用户角色、用户文本和聊天历史
def clear(user_role, user_text, chatbot):
    return None, None, []

# 函数：获取指定 AI 角色对应的图像并返回
def get_image(ai_role):
    # 获取 AI 角色对应的英文名
    role_en = NAME_DICT[ai_role]
    # 打开并返回指定路径下的图像文件
    return Image.open(f'images/{role_en}.jpg'), None, None, []

# 使用 gr.Blocks() 创建 Gradio 交互式界面的代码块
with gr.Blocks() as demo:
    # 这里添加后续的 Gradio 交互式界面相关代码
    gr.Markdown(
        """
        # Chat凉宫春日 ChatHaruhi
        ## Reviving Anime Character in Reality via Large Language Model

        ChatHaruhi2.0的demo implemented by [chenxi](https://github.com/todochenxi)

        更多信息见项目github链接 [https://github.com/LC1332/Chat-Haruhi-Suzumiya](https://github.com/LC1332/Chat-Haruhi-Suzumiya)

        如果觉得有趣请拜托为我们点上star. If you find it interesting, please be kind enough to give us a star.

        user_role 为角色扮演的人物 请尽量设置为与剧情相关的人物 且不要与主角同名
        """
    )
    # 创建一个包含多行文本的 Markdown 组件，用于展示关于 ChatHaruhi 项目的信息和提示

    with gr.Row():
        # 创建一个横向排列的组件行

        chatbot = gr.Chatbot()
        # 创建一个聊天机器人组件

        role_image = gr.Image(height=400, value="./images/haruhi.jpg")
        # 创建一个图片组件，用于显示角色的图像，并设置高度为 400 像素，图像路径为 './images/haruhi.jpg'

    with gr.Row():
        # 创建第二个横向排列的组件行

        user_role = gr.Textbox(label="user_role")
        # 创建一个文本框组件，用于输入角色扮演的人物名称，标签为 "user_role"

        user_text = gr.Textbox(label="user_text")
        # 创建一个文本框组件，用于输入用户的文本，标签为 "user_text"

    with gr.Row():
        # 创建第三个横向排列的组件行

        submit = gr.Button("Submit")
        # 创建一个提交按钮组件，显示文本为 "Submit"

        clean = gr.ClearButton(value="Clear")
        # 创建一个清除按钮组件，显示文本为 "Clear"

    ai_role = gr.Radio(['汤师爷', '慕容复', '李云龙',
                        'Luna', '王多鱼', 'Ron', '鸠摩智',
                        'Snape', '凉宫春日', 'Malfoy', '虚竹',
                        '萧峰', '段誉', 'Hermione', 'Dumbledore',
                        '王语嫣',
                        'Harry', 'McGonagall',
                        '白展堂', '佟湘玉', '郭芙蓉',
                        '旅行者', '钟离', '胡桃',
                        'Sheldon', 'Raj', 'Penny',
                        '韦小宝', '乔峰', '神里绫华',
                        '雷电将军', '于谦'], label="characters", value='凉宫春日')
    # 创建一个单选按钮组件，用于选择角色，选项包括多个角色名称，初始选择为 '凉宫春日'

    ai_role.change(get_image, ai_role, [role_image, user_role, user_text, chatbot])
    # 当单选按钮的选择发生变化时，调用 get_image 函数，并传递角色图像、用户角色、用户文本和聊天机器人作为参数

    user_text.submit(fn=respond, inputs=[user_role, user_text, ai_role, chatbot], outputs=[chatbot, user_text])
    # 当用户文本框内按下提交时，调用 respond 函数，传递用户角色、用户文本、AI角色和聊天机器人作为输入，并将聊天机器人和用户文本框作为输出

    submit.click(fn=respond, inputs=[user_role, user_text, ai_role, chatbot], outputs=[chatbot, user_text])
    # 当提交按钮被点击时，调用 respond 函数，传递用户角色、用户文本、AI角色和聊天机器人作为输入，并将聊天机器人和用户文本框作为输出

    clean.click(clear, [user_role, user_text, chatbot], [user_role, user_text, chatbot])
    # 当清除按钮被点击时，调用 clear 函数，传递用户角色、用户文本和聊天机器人作为输入和输出
# 调用名为 demo 的对象或函数的 launch 方法，并传入两个关键字参数 debug 和 share，分别设置为 True
demo.launch(debug=True, share=True)
```