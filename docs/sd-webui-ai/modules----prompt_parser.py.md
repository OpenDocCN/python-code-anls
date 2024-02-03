# `stable-diffusion-webui\modules\prompt_parser.py`

```py
# 导入必要的模块
from __future__ import annotations

import re
from collections import namedtuple
import lark

# 定义一个语法解析器，用于解析输入的字符串
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

# 根据给定的提示、基础步数和高分辨率步数，获取学习条件提示的时间表
def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]

    # 返回学习条件提示的时间表
    # 测试函数 g 的输入和输出
    >>> g("((a][:b:c [d:3]")
    # 返回结果列表，包含每个子字符串的起始位置和子字符串本身
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    # 返回结果列表，包含每个子字符串的起始位置和子字符串本身
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    >>> g("[fe|]male")
    # 返回结果列表，包含每个子字符串的起始位置和子字符串本身
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("[fe|||]male")
    # 返回结果列表，包含每个子字符串的起始位置和子字符串本身
    [[1, 'female'], [2, 'male'], [3, 'male'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'male'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10)[0]
    # 使用 lambda 函数定义 g，调用 get_learned_conditioning_prompt_schedules 函数
    >>> g("a [b:.5] c")
    # 返回结果列表，包含每个子字符串的起始位置和子字符串本身
    [[10, 'a b c']]
    >>> g("a [b:1.5] c")
    # 返回结果列表，包含每个子字符串的起始位置和子字符串本身
    [[5, 'a  c'], [10, 'a b c']]
    """

    # 如果 hires_steps 为 None 或者 use_old_scheduling 为真，则设置偏移量和步数
    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    # 定义函数 collect_steps，用于收集步数
    def collect_steps(steps, tree):
        res = [steps]

        # 定义类 CollectSteps，继承自 lark.Visitor
        class CollectSteps(lark.Visitor):
            # 处理 scheduled 规则
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                # 根据 use_old_scheduling 的值调整步数
                if use_old_scheduling:
                    v = v*steps if v<1 else v
                else:
                    if "." in s:
                        v = (v - flt_offset) * steps
                    else:
                        v = (v - int_offset)
                # 更新树中的步数值
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            # 处理 alternate 规则
            def alternate(self, tree):
                # 将范围内的步数添加到结果列表中
                res.extend(range(1, steps+1))

        # 实例化 CollectSteps 类并访问树
        CollectSteps().visit(tree)
        # 返回排序后的结果列表
        return sorted(set(res))
    # 定义一个函数，根据给定的步骤和语法树进行处理
    def at_step(step, tree):
        # 定义一个继承自 lark.Transformer 的类 AtStep
        class AtStep(lark.Transformer):
            # 处理 scheduled 规则
            def scheduled(self, args):
                before, after, _, when, _ = args
                # 如果当前步骤小于等于规定的步骤时，返回 before，否则返回 after
                yield before or () if step <= when else after
            # 处理 alternate 规则
            def alternate(self, args):
                # 如果参数为空，则替换为空字符串
                args = ["" if not arg else arg for arg in args]
                # 返回在参数列表中根据当前步骤取模后的值
                yield args[(step - 1) % len(args)]
            # 处理 start 规则
            def start(self, args):
                # 定义一个函数，用于将嵌套的列表展开为字符串
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                # 返回展开后的字符串
                return ''.join(flatten(args))
            # 处理 plain 规则
            def plain(self, args):
                # 返回参数列表中第一个元素的值
                yield args[0].value
            # 处理默认情况
            def __default__(self, data, children, meta):
                # 遍历子节点并返回
                for child in children:
                    yield child
        # 返回 AtStep 类的实例对给定的语法树进行转换
        return AtStep().transform(tree)

    # 定义一个函数，根据给定的提示获取调度信息
    def get_schedule(prompt):
        try:
            # 解析提示并得到语法树
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            # 如果解析出错，则返回包含步骤和提示的列表
            if 0:
                import traceback
                traceback.print_exc()
            return [[steps, prompt]]
        # 返回包含步骤和处理后的语法树的列表
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    # 使用集合去除重复的提示，构建提示字典
    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    # 返回根据提示获取的调度信息列表
    return [promptdict[prompt] for prompt in prompts]
# 使用 namedtuple 创建一个命名元组 ScheduledPromptConditioning，包含 end_at_step 和 cond 两个字段
ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])

# 定义一个类 SdConditioning，继承自 list，用于存储 stable diffusion 模型的条件提示
class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    # 初始化方法，接受 prompts、is_negative_prompt、width、height 和 copy_from 参数
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        # 将 prompts 中的内容添加到当前列表中
        self.extend(prompts)

        # 如果 copy_from 为 None，则将其设置为 prompts
        if copy_from is None:
            copy_from = prompts

        # 设置 is_negative_prompt 属性为传入的 is_negative_prompt 参数或者 copy_from 中的 is_negative_prompt 属性
        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        # 设置 width 属性为传入的 width 参数或者 copy_from 中的 width 属性
        self.width = width or getattr(copy_from, 'width', None)
        # 设置 height 属性为传入的 height 参数或者 copy_from 中的 height 属性
        self.height = height or getattr(copy_from, 'height', None)

# 定义一个函数 get_learned_conditioning，用于将提示列表转换为提示计划列表
def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    # 初始化结果列表
    res = []
    # 获取学习到的条件提示的调度表
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    # 创建一个空的缓存字典
    cache = {}
    
    # 遍历每个提示和对应的调度表
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
    
        # 从缓存中获取已经计算过的结果
        cached = cache.get(prompt, None)
        # 如果已经计算过，则直接添加到结果列表中并继续下一个循环
        if cached is not None:
            res.append(cached)
            continue
    
        # 从提示调度表中提取文本
        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        # 获取学习到的条件
        conds = model.get_learned_conditioning(texts)
    
        # 创建条件调度表
        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            # 根据条件类型不同，构建不同的条件对象
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]
    
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
    
        # 将计算结果添加到缓存中
        cache[prompt] = cond_schedule
        # 将计算结果添加到结果列表中
        res.append(cond_schedule)
    
    # 返回最终结果列表
    return res
# 编译正则表达式，用于匹配单词 "AND"
re_AND = re.compile(r"\bAND\b")
# 编译正则表达式，用于匹配带有权重的文本
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")

# 获取多条件提示列表的函数，参数为 SdConditioning 类型或字符串列表
def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    # 用于存储结果的索引列表
    res_indexes = []

    # 用于存储提示文本和对应索引的字典
    prompt_indexes = {}
    # 创建 SdConditioning 对象并清空
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    # 遍历每个提示
    for prompt in prompts:
        # 使用 "AND" 分隔符拆分提示文本
        subprompts = re_AND.split(prompt)

        indexes = []
        # 遍历拆分后的子提示
        for subprompt in subprompts:
            # 匹配带有权重的文本
            match = re_weight.search(subprompt)

            # 获取文本和权重，如果没有匹配到则默认权重为 1.0
            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            # 将权重转换为浮点数，如果没有权重则默认为 1.0
            weight = float(weight) if weight is not None else 1.0

            # 获取文本在字典中的索引，如果不存在则添加到列表中
            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    # 返回结果索引列表、提示文本列表和提示索引字典
    return res_indexes, prompt_flat_list, prompt_indexes

# 可组合的定时提示条件类
class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight

# 多条件学习条件类
class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # 需要发送到 DDIM/PLMS 的形状字段
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch

# 获取多条件学习条件的函数，返回 MulticondLearnedConditioning 对象
def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False) -> MulticondLearnedConditioning:
    """同 get_learned_conditioning，但返回每个提示的 ScheduledPromptConditioning 列表以及每个提示的权重对象。
    对于每个提示，通过使用 AND 分隔符来获取列表。

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """
    # 调用函数获取多条件提示列表的索引、扁平化的提示列表和提示索引
    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    # 调用函数获取学习到的条件
    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, hires_steps, use_old_scheduling)

    # 初始化结果列表
    res = []
    # 遍历索引列表
    for indexes in res_indexes:
        # 将学习到的条件和权重组成 ComposableScheduledPromptConditioning 对象，并添加到结果列表中
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

    # 返回多条件学习到的条件对象
    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)
# 定义一个继承自 dict 的类 DictWithShape
class DictWithShape(dict):
    # 初始化方法，接受一个字典 x 和一个 shape 参数
    def __init__(self, x, shape):
        # 调用父类的初始化方法
        super().__init__()
        # 更新字典内容为 x
        self.update(x)

    # 定义 shape 属性，返回字典中 "crossattn" 键对应值的形状
    @property
    def shape(self):
        return self["crossattn"].shape


# 重构条件批次函数，接受一个列表 c 和当前步骤 current_step
def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    # 获取第一个条件的参数
    param = c[0][0].cond
    # 判断参数是否为字典类型
    is_dict = isinstance(param, dict)

    # 如果参数是字典类型
    if is_dict:
        # 将参数赋值给 dict_cond
        dict_cond = param
        # 创建一个字典 res，键为 dict_cond 的键，值为 torch.zeros 函数生成的张量
        res = {k: torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype) for k, param in dict_cond.items()}
        # 使用 DictWithShape 类创建一个新的字典 res
        res = DictWithShape(res, (len(c),) + dict_cond['crossattn'].shape)
    # 如果参数不是字典类型
    else:
        # 创建一个张量 res，形状为 (len(c),) + param.shape
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)

    # 遍历条件列表 c
    for i, cond_schedule in enumerate(c):
        target_index = 0
        # 遍历当前条件的每个条目
        for current, entry in enumerate(cond_schedule):
            # 如果当前步骤小于等于条目的结束步骤
            if current_step <= entry.end_at_step:
                target_index = current
                break

        # 如果参数是字典类型
        if is_dict:
            # 遍历当前条件的参数字典
            for k, param in cond_schedule[target_index].cond.items():
                # 将参数赋值给 res 中对应键的张量
                res[k][i] = param
        # 如果参数不是字典类型
        else:
            # 将当前条件的参数赋值给 res 的第 i 个张量
            res[i] = cond_schedule[target_index].cond

    # 返回结果 res
    return res


# 堆叠条件函数，接受一个张量列表 tensors
def stack_conds(tensors):
    # 获取张量中最大的 token 数量
    token_count = max([x.shape[0] for x in tensors])
    # 遍历张量列表
    for i in range(len(tensors)):
        # 如果张量的 token 数量不等于最大 token 数量
        if tensors[i].shape[0] != token_count:
            # 获取最后一个向量
            last_vector = tensors[i][-1:]
            # 将最后一个向量重复多次，使得张量的 token 数量与最大 token 数量相等
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            # 将重复的向量堆叠到原始张量中
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])

    # 返回堆叠后的张量
    return torch.stack(tensors)


# 重构多条件批次函数，接受一个 MulticondLearnedConditioning 对象 c 和当前步骤 current_step
def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    # 获取批次中第一个条件的第一个计划的参数
    param = c.batch[0][0].schedules[0].cond

    # 初始化一个空列表 tensors 和 conds_list
    tensors = []
    conds_list = []
    # 遍历批次中的可组合提示
    for composable_prompts in c.batch:
        # 存储每个批次的条件
        conds_for_batch = []

        # 遍历每个可组合提示
        for composable_prompt in composable_prompts:
            # 初始化目标索引为0
            target_index = 0
            # 遍历当前可组合提示的计划
            for current, entry in enumerate(composable_prompt.schedules):
                # 如果当前步骤小于等于计划的结束步骤，则更新目标索引并跳出循环
                if current_step <= entry.end_at_step:
                    target_index = current
                    break

            # 将条件的长度和权重添加到条件列表中
            conds_for_batch.append((len(tensors), composable_prompt.weight))
            # 将当前目标索引对应的条件添加到张量列表中
            tensors.append(composable_prompt.schedules[target_index].cond)

        # 将当前批次的条件列表添加到总条件列表中
        conds_list.append(conds_for_batch)

    # 如果第一个张量是字典
    if isinstance(tensors[0], dict):
        # 获取字典的键列表
        keys = list(tensors[0].keys())
        # 对字典中的值进行堆叠操作
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        # 创建具有形状的字典
        stacked = DictWithShape(stacked, stacked['crossattn'].shape)
    else:
        # 对张量进行堆叠操作，并指定设备和数据类型
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)

    # 返回条件列表和堆叠后的张量
    return conds_list, stacked
# 编译正则表达式，用于匹配注意力标记
re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

# 编译正则表达式，用于匹配 BREAK 标记
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

# 解析带有注意力标记的字符串，返回文本和其关联权重的列表
def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    # 初始化结果列表、圆括号列表和方括号列表
    res = []
    round_brackets = []
    square_brackets = []

    # 圆括号乘数和方括号乘数
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    # 定义函数，用于对指定位置之后的文本和权重进行乘法操作
    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier
    # 遍历正则表达式匹配到的所有关注文本
    for m in re_attention.finditer(text):
        # 获取匹配到的文本
        text = m.group(0)
        # 获取匹配到的权重
        weight = m.group(1)

        # 如果文本以 '\' 开头
        if text.startswith('\\'):
            # 将文本和权重添加到结果列表中
            res.append([text[1:], 1.0])
        # 如果文本为 '('
        elif text == '(':
            # 记录当前位置到圆括号列表中
            round_brackets.append(len(res))
        # 如果文本为 '['
        elif text == '[':
            # 记录当前位置到方括号列表中
            square_brackets.append(len(res))
        # 如果存在权重且圆括号列表不为空
        elif weight is not None and round_brackets:
            # 对圆括号内的文本进行权重乘法
            multiply_range(round_brackets.pop(), float(weight))
        # 如果文本为 ')' 且圆括号列表不为空
        elif text == ')' and round_brackets:
            # 对圆括号内的文本进行权重乘法
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        # 如果文本为 ']' 且方括号列表不为空
        elif text == ']' and square_brackets:
            # 对方括号内的文本进行权重乘法
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            # 将文本按照断点正则表达式分割
            parts = re.split(re_break, text)
            # 遍历分割后的部分
            for i, part in enumerate(parts):
                # 如果不是第一个部分
                if i > 0:
                    # 添加断点标记到结果列表中
                    res.append(["BREAK", -1])
                # 添加部分文本和权重到结果列表中
                res.append([part, 1.0])

    # 对剩余的圆括号内的文本进行权重乘法
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    # 对剩余的方括号内的文本进行权重乘法
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    # 如果结果列表为空
    if len(res) == 0:
        # 将空字符串和权重 1.0 添加到结果列表中
        res = [["", 1.0]]

    # 合并相同权重的文本
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    # 返回处理后的结果列表
    return res
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 导入 doctest 模块，用于测试文档中的示例代码
    import doctest
    # 运行 doctest 模块中的测试用例，设置选项为规范化空白字符
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
# 如果当前脚本被导入，则执行以下代码块
else:
    # 导入 torch 模块，用于其他用途，例如提高 doctest 的速度
    import torch  # doctest faster
```