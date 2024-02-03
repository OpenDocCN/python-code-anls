# `stable-diffusion-webui\modules\styles.py`

```py
# 导入必要的模块
import csv
import fnmatch
import os
import os.path
import typing
import shutil

# 定义一个命名元组，包含样式名称、提示信息、否定提示信息和路径
class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str
    path: str = None

# 合并样式提示信息和提示信息
def merge_prompts(style_prompt: str, prompt: str) -> str:
    if "{prompt}" in style_prompt:
        res = style_prompt.replace("{prompt}", prompt)
    else:
        parts = filter(None, (prompt.strip(), style_prompt.strip()))
        res = ", ".join(parts)

    return res

# 将样式应用到提示信息
def apply_styles_to_prompt(prompt, styles):
    for style in styles:
        prompt = merge_prompts(style, prompt)

    return prompt

# 从提示信息中解析出样式文本
def unwrap_style_text_from_prompt(style_text, prompt):
    """
    Checks the prompt to see if the style text is wrapped around it. If so,
    returns True plus the prompt text without the style text. Otherwise, returns
    False with the original prompt.

    Note that the "cleaned" version of the style text is only used for matching
    purposes here. It isn't returned; the original style text is not modified.
    """
    stripped_prompt = prompt
    stripped_style_text = style_text
    if "{prompt}" in stripped_style_text:
        # Work out whether the prompt is wrapped in the style text. If so, we
        # return True and the "inner" prompt text that isn't part of the style.
        try:
            left, right = stripped_style_text.split("{prompt}", 2)
        except ValueError as e:
            # If the style text has multple "{prompt}"s, we can't split it into
            # two parts. This is an error, but we can't do anything about it.
            print(f"Unable to compare style text to prompt:\n{style_text}")
            print(f"Error: {e}")
            return False, prompt
        if stripped_prompt.startswith(left) and stripped_prompt.endswith(right):
            prompt = stripped_prompt[len(left) : len(stripped_prompt) - len(right)]
            return True, prompt
    else:
        # 如果给定的提示以样式文本结尾，则返回 True 和截取样式文本之前的提示文本
        if stripped_prompt.endswith(stripped_style_text):
            # 截取提示文本直到样式文本开始的位置
            prompt = stripped_prompt[: len(stripped_prompt) - len(stripped_style_text)]
            # 如果提示文本以逗号和空格结尾，则去除逗号和空格
            if prompt.endswith(", "):
                prompt = prompt[:-2]
            # 返回 True 和截取后的提示文本
            return True, prompt

    # 如果不符合上述条件，则返回 False 和原始提示文本
    return False, prompt
# 提取原始提示信息
def extract_original_prompts(style: PromptStyle, prompt, negative_prompt):
    """
    Takes a style and compares it to the prompt and negative prompt. If the style
    matches, returns True plus the prompt and negative prompt with the style text
    removed. Otherwise, returns False with the original prompt and negative prompt.
    """
    # 如果样式的提示和负面提示都为空，则返回原始的提示和负面提示
    if not style.prompt and not style.negative_prompt:
        return False, prompt, negative_prompt

    # 从提示中提取样式文本，与样式进行比较
    match_positive, extracted_positive = unwrap_style_text_from_prompt(
        style.prompt, prompt
    )
    # 如果样式文本匹配失败，则返回原始的提示和负面提示
    if not match_positive:
        return False, prompt, negative_prompt

    # 从负面提示中提取样式文本，与样式进行比较
    match_negative, extracted_negative = unwrap_style_text_from_prompt(
        style.negative_prompt, negative_prompt
    )
    # 如果样式文本匹配失败，则返回原始的提示和负面提示
    if not match_negative:
        return False, prompt, negative_prompt

    # 返回匹配成功的标志以及提取后的提示和负面提示
    return True, extracted_positive, extracted_negative


# 样式数据库类
class StyleDatabase:
    # 初始化方法
    def __init__(self, path: str):
        # 创建一个没有样式的默认样式对象
        self.no_style = PromptStyle("None", "", "", None)
        # 初始化样式字典和路径
        self.styles = {}
        self.path = path

        # 拆分路径，获取文件名和扩展名
        folder, file = os.path.split(self.path)
        filename, _, ext = file.partition('*')
        # 设置默认路径
        self.default_path = os.path.join(folder, filename + ext)

        # 获取除了路径之外的所有字段
        self.prompt_fields = [field for field in PromptStyle._fields if field != "path"]

        # 重新加载样式数据库
        self.reload()
    def reload(self):
        """
        Clears the style database and reloads the styles from the CSV file(s)
        matching the path used to initialize the database.
        """
        # 清空样式数据库
        self.styles.clear()

        # 拆分路径和文件名
        path, filename = os.path.split(self.path)

        # 如果文件名中包含通配符"*"
        if "*" in filename:
            # 构建文件名通配符
            fileglob = filename.split("*")[0] + "*.csv"
            filelist = []
            # 遍历路径下的文件
            for file in os.listdir(path):
                # 匹配文件名通配符
                if fnmatch.fnmatch(file, fileglob):
                    filelist.append(file)
                    # 添加可见的样式列表分隔符
                    half_len = round(len(file) / 2)
                    divider = f"{'-' * (20 - half_len)} {file.upper()}"
                    divider = f"{divider} {'-' * (40 - len(divider))}"
                    # 将分隔符作为键，创建一个新的 PromptStyle 对象
                    self.styles[divider] = PromptStyle(
                        f"{divider}", None, None, "do_not_save"
                    )
                    # 从这个CSV文件中加载样式
                    self.load_from_csv(os.path.join(path, file))
            # 如果没有找到匹配的文件
            if len(filelist) == 0:
                print(f"No styles found in {path} matching {fileglob}")
                return
        # 如果路径不存在
        elif not os.path.exists(self.path):
            print(f"Style database not found: {self.path}")
            return
        else:
            # 从指定路径加载CSV文件中的样式
            self.load_from_csv(self.path)
    def load_from_csv(self, path: str):
        # 打开 CSV 文件，使用 utf-8-sig 编码读取，跳过开头空格
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            # 创建 CSV 字典读取器
            reader = csv.DictReader(file, skipinitialspace=True)
            # 遍历 CSV 文件的每一行
            for row in reader:
                # 忽略空行或以注释符号开头的行
                if not row or row["name"].startswith("#"):
                    continue
                # 支持加载旧的 CSV 格式，包含 "name, text" 列
                prompt = row["prompt"] if "prompt" in row else row["text"]
                negative_prompt = row.get("negative_prompt", "")
                # 将样式添加到数据库中
                self.styles[row["name"]] = PromptStyle(
                    row["name"], prompt, negative_prompt, path
                )

    def get_style_paths(self) -> set:
        """返回所有加载样式的文件路径的集合"""
        # 更新没有路径的样式到默认路径
        for style in list(self.styles.values()):
            if not style.path:
                self.styles[style.name] = style._replace(path=self.default_path)

        # 创建包括默认路径在内的所有不同路径的列表
        style_paths = set()
        style_paths.add(self.default_path)
        for _, style in self.styles.items():
            if style.path:
                style_paths.add(style.path)

        # 移除只是列表分隔符的样式路径
        style_paths.discard("do_not_save")

        return style_paths

    def get_style_prompts(self, styles):
        # 返回样式对应的提示列表
        return [self.styles.get(x, self.no_style).prompt for x in styles]

    def get_negative_style_prompts(self, styles):
        # 返回样式对应的负面提示列表
        return [self.styles.get(x, self.no_style).negative_prompt for x in styles]

    def apply_styles_to_prompt(self, prompt, styles):
        # 将样式应用到提示上
        return apply_styles_to_prompt(
            prompt, [self.styles.get(x, self.no_style).prompt for x in styles]
        )
    # 将负面样式应用到提示文本中
    def apply_negative_styles_to_prompt(self, prompt, styles):
        # 调用函数 apply_styles_to_prompt，将负面样式应用到提示文本中
        return apply_styles_to_prompt(
            prompt, [self.styles.get(x, self.no_style).negative_prompt for x in styles]
        )

    # 保存样式信息到文件
    def save_styles(self, path: str = None) -> None:
        # 路径参数已弃用，但为了向后兼容性而保留
        _ = path

        # 获取样式文件路径
        style_paths = self.get_style_paths()

        # 获取样式文件名列表
        csv_names = [os.path.split(path)[1].lower() for path in style_paths]

        # 遍历样式文件路径
        for style_path in style_paths:
            # 如果样式文件存在，则创建备份文件
            if os.path.exists(style_path):
                shutil.copy(style_path, f"{style_path}.bak")

            # 将样式写入 CSV 文件
            with open(style_path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.prompt_fields)
                writer.writeheader()
                # 遍历样式字典，将样式写入文件
                for style in (s for s in self.styles.values() if s.path == style_path):
                    # 跳过样式列表分隔符，例如 "STYLES.CSV"
                    if style.name.lower().strip("# ") in csv_names:
                        continue
                    # 写入样式字段，忽略路径字段
                    writer.writerow(
                        {k: v for k, v in style._asdict().items() if k != "path"}
                    )
    # 从用户输入的提示和负面提示中提取样式信息
    def extract_styles_from_prompt(self, prompt, negative_prompt):
        # 初始化一个空列表用于存储提取出的样式信息
        extracted = []

        # 获取所有可用的样式列表
        applicable_styles = list(self.styles.values())

        # 进入循环，直到没有样式匹配为止
        while True:
            # 初始化一个变量用于存储找到的样式
            found_style = None

            # 遍历所有可用的样式
            for style in applicable_styles:
                # 调用函数从样式中提取原始提示信息，并返回是否匹配、新的提示和新的负面提示
                is_match, new_prompt, new_neg_prompt = extract_original_prompts(
                    style, prompt, negative_prompt
                )
                # 如果匹配成功
                if is_match:
                    found_style = style
                    prompt = new_prompt
                    negative_prompt = new_neg_prompt
                    break

            # 如果没有找到匹配的样式，则退出循环
            if not found_style:
                break

            # 从可用样式列表中移除已经找到的样式
            applicable_styles.remove(found_style)
            # 将找到的样式名称添加到提取列表中
            extracted.append(found_style.name)

        # 返回提取出的样式列表（倒序）、新的提示和新的负面提示
        return list(reversed(extracted)), prompt, negative_prompt
```