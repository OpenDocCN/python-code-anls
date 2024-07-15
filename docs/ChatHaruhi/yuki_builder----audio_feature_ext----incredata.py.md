# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\incredata.py`

```py
# 导入操作系统相关的模块
import os
# 导入文件和目录操作相关的模块
import shutil
# 从工具模块中导入指定的函数
from tool import get_filename, read_tolist, get_first_subdir

# 定义一个类 IncrementData，用于处理增量数据
class IncrementData():
    def __init__(self, audio_out_dir, audio_roles_dir, srt_out_dir):
        # 初始化类实例时，传入音频输出目录、音频角色目录和字幕输出目录
        self.audio_out_dir = audio_out_dir
        self.audio_roles_dir = audio_roles_dir
        self.srt_out_dir = srt_out_dir

    # 定义一个方法 static_origin，用于统计原始数据
    def static_origin(self):
        # 初始化统计字典
        stattics_dic = {}
        # 获取音频角色目录下的第一层子目录列表
        role_lis = get_first_subdir(self.audio_roles_dir)
        # 遍历角色列表
        for sub_dir in role_lis:
            # 获取角色名，这里假设角色目录使用斜杠分隔路径，取最后一个斜杠后的部分作为角色名
            role = sub_dir.split('/')[-1]
            # 获取角色目录下的文件名列表
            lis = get_filename(sub_dir)
            # 将角色名和文件名列表关联存入统计字典
            stattics_dic[role] = [item[0] for item in get_filename(sub_dir)]

        # 将统计字典中所有文件名列表合并为一个原始文件名列表
        origin_lis = [item for sub_lis in list(stattics_dic.values()) for item in sub_lis]
        # 返回合并后的原始文件名列表
        return origin_lis
    def process(self):
        # 获取输出目录中名为 'annotate.txt' 的文件路径列表
        golden_res = get_filename(self.srt_out_dir, 'annotate.txt')
        # 初始化空列表，用于存放相同文件名的音频路径
        same_lis = []
        # 初始化计数器变量 i 和 j
        i = 0
        j = 0
        # 遍历 golden_res 中的每个文件名及其路径
        for file, pth in golden_res[:]:
            # 将文件内容读取为列表
            srt_lis = read_tolist(pth)
            # 提取文件名中的前两部分并用下划线连接，作为新的文件名
            file_name = '_'.join(file.split('_')[:2])
            # 初始化空字典 annote_dic，用于存放角色和文本的对应关系
            annote_dic = {}
            # 遍历 srt_lis 中的每一行文本
            for line in srt_lis:
                # 将每行文本按照 ":「" 进行分割，得到角色和文本
                role, text = line.split(":「")
                # 去掉文本末尾的双引号
                text = text[:-1]
                # 如果文本不在 annote_dic 中，将文本作为键，角色作为值存入字典
                if text not in annote_dic:
                    annote_dic[text] = [role]
                # 如果文本已经在 annote_dic 中，则将角色追加到值列表中
                else:
                    annote_dic[text].append(role)
            # 筛选出只出现一次的文本-角色对应关系，形成新的字典 real_dic
            real_dic = {k: v[0] for k, v in annote_dic.items() if len(v) == 1}
            # 构建对应的音频目录路径
            corres_dir = os.path.join(self.audio_out_dir, f'{file_name}/voice')
            # 获取目录中的文件名及其路径列表
            audio_lis = get_filename(corres_dir)

            # 初始化空字典 audio_dic，用于存放音频文本和对应文件信息的关系
            audio_dic = {}
            # 遍历 audio_lis 中的每个音频文件名及其路径
            for aud_name, aud_pth in audio_lis:
                # 提取文件名中除去扩展名后的部分作为音频文本
                file_text = os.path.splitext(aud_name)[0]
                audio_text = ''.join(file_text.split('_')[1:])
                # 如果音频文本不在 audio_dic 中，将音频文本作为键，文件信息作为值存入字典
                if audio_text not in audio_dic:
                    audio_dic[audio_text] = [[aud_name, aud_pth]]
                # 如果音频文本已经在 audio_dic 中，则将文件信息追加到值列表中
                else:
                    audio_dic[audio_text].append([aud_name, aud_pth])
            # 筛选出只出现一次的音频文本-文件信息对应关系，形成新的字典 new_audio_dic
            new_audio_dic = {k: v[0] for k, v in audio_dic.items() if len(v) == 1}
            # 遍历 new_audio_dic 中的每个音频文本及其对应的文件信息
            for audio_text, value in new_audio_dic.items():
                aud_name, aud_pth = value
                # 如果音频文本在 real_dic 中存在对应关系
                if audio_text in real_dic:
                    # 获取音频文本对应的角色
                    role = real_dic[audio_text]
                    # 构建目标音频角色目录路径
                    new_aud_dir = os.path.join(self.audio_roles_dir, role)
                    # 如果目录不存在，则创建之
                    os.makedirs(new_aud_dir, exist_ok=True)
                    # 构建目标音频文件路径
                    new_aud_pth = os.path.join(new_aud_dir, aud_name)
                    
                    # 拷贝音频文件到目标路径，如果目标路径文件不存在
                    if not os.path.exists(new_aud_pth):
                        shutil.copy(aud_pth, new_aud_pth)
                        # 增加计数器 i 的值，并打印拷贝信息
                        i += 1
                        print(f'{role} + 1 {aud_name},{i}')
                        pass
                    # 如果目标路径文件已存在，则不执行拷贝操作
                    elif os.path.exists(new_aud_pth):
                        pass

        # 计算与 self.origin_lis 不同的元素列表 chaji_lis
        # print(chaji_lis)
```