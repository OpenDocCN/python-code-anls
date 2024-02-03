# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\profiler.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制

# 导入 sys 模块
import sys
# 导入 paddle 模块
import paddle

# 用于记录分析器函数调用次数的全局变量
# 用于指定训练步骤的跟踪范围
_profiler_step_id = 0

# 用于避免每次都从字符串解析的全局变量
_profiler_options = None

class ProfilerOptions(object):
    '''
    使用字符串初始化 ProfilerOptions。
    字符串应该采用以下格式: "key1=value1;key2=value;key3=value3"。
    例如:
      "profile_path=model.profile"
      "batch_range=[50, 60]; profile_path=model.profile"
      "batch_range=[50, 60]; tracer_option=OpDetail; profile_path=model.profile"
    ProfilerOptions 支持以下键值对:
      batch_range      - 一个整数列表，例如 [100, 110]。
      state            - 一个字符串，可选值为 'CPU'、'GPU' 或 'All'。
      sorted_key       - 一个字符串，可选值为 'calls'、'total'、'max'、'min' 或 'ave'。
      tracer_option    - 一个字符串，可选值为 'Default'、'OpDetail'、'AllOpDetail'。
      profile_path     - 一个字符串，保存序列化的分析数据的路径，
                         可用于生成时间线。
      exit_on_finished - 一个布尔值。
    '''
    # 初始化函数，接受一个字符串参数作为选项设置
    def __init__(self, options_str):
        # 断言参数是字符串类型
        assert isinstance(options_str, str)
    
        # 初始化默认选项字典
        self._options = {
            'batch_range': [10, 20],
            'state': 'All',
            'sorted_key': 'total',
            'tracer_option': 'Default',
            'profile_path': '/tmp/profile',
            'exit_on_finished': True
        }
        # 解析传入的选项字符串
        self._parse_from_string(options_str)
    
    # 从字符串解析选项设置
    def _parse_from_string(self, options_str):
        # 去除空格并按分号分割键值对
        for kv in options_str.replace(' ', '').split(';'):
            # 拆分键值对
            key, value = kv.split('=')
            # 根据键值对的不同情况进行处理
            if key == 'batch_range':
                # 处理批量范围的值
                value_list = value.replace('[', '').replace(']', '').split(',')
                value_list = list(map(int, value_list))
                if len(value_list) >= 2 and value_list[0] >= 0 and value_list[1] > value_list[0]:
                    self._options[key] = value_list
            elif key == 'exit_on_finished':
                # 处理退出完成时的值
                self._options[key] = value.lower() in ("yes", "true", "t", "1")
            elif key in ['state', 'sorted_key', 'tracer_option', 'profile_path']:
                # 处理状态、排序键、追踪器选项、配置文件路径的值
                self._options[key] = value
    
    # 获取选项字典中指定键的值
    def __getitem__(self, name):
        # 如果选项字典中不存在指定键，则抛出异常
        if self._options.get(name, None) is None:
            raise ValueError("ProfilerOptions does not have an option named %s." % name)
        # 返回指定键的值
        return self._options[name]
# 定义一个函数，用于启用基于 PaddlePaddle 的分析器进行操作级别的计时
# 分析器使用一个独立的变量来计算分析器步骤，每次调用此函数被视为一个分析器步骤
def add_profiler_step(options_str=None):
    # 如果未提供分析器选项字符串，则直接返回
    if options_str is None:
        return

    # 声明全局变量_profiler_step_id和_profiler_options
    global _profiler_step_id
    global _profiler_options

    # 如果_profiler_options为None，则使用options_str初始化ProfilerOptions对象
    if _profiler_options is None:
        _profiler_options = ProfilerOptions(options_str)

    # 如果当前_profiler_step_id等于_profiler_options中的batch_range的第一个值
    # 则启动分析器
    if _profiler_step_id == _profiler_options['batch_range'][0]:
        paddle.utils.profiler.start_profiler(_profiler_options['state'],
                                             _profiler_options['tracer_option'])
    # 如果当前_profiler_step_id等于_profiler_options中的batch_range的第二个值
    # 则停止分析器，并根据设置输出分析结果
    elif _profiler_step_id == _profiler_options['batch_range'][1]:
        paddle.utils.profiler.stop_profiler(_profiler_options['sorted_key'],
                                            _profiler_options['profile_path'])
        # 如果设置了在完成后退出，则调用sys.exit(0)退出程序
        if _profiler_options['exit_on_finished']:
            sys.exit(0)

    # 增加_profiler_step_id以便下一次调用时能正确判断是否需要启动或停止分析器
    _profiler_step_id += 1
```