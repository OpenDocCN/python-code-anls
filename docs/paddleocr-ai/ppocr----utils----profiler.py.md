# `.\PaddleOCR\ppocr\utils\profiler.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制

# 导入 sys 模块
import sys
# 导入 paddle.profiler 模块并重命名为 profiler
import paddle.profiler as profiler

# 用于记录 profiler 函数调用次数的全局变量
# 用于指定训练步骤的跟踪范围
_profiler_step_id = 0

# 用于避免每次都从字符串中解析的全局变量
_profiler_options = None
_prof = None

class ProfilerOptions(object):
    '''
    使用字符串初始化 ProfilerOptions。
    字符串应该是以下格式："key1=value1;key2=value;key3=value3"。
    例如：
      "profile_path=model.profile"
      "batch_range=[50, 60]; profile_path=model.profile"
      "batch_range=[50, 60]; tracer_option=OpDetail; profile_path=model.profile"

    ProfilerOptions 支持以下键值对：
      batch_range      - 一个整数列表，例如 [100, 110]。
      state            - 一个字符串，可选值为 'CPU'、'GPU' 或 'All'。
      sorted_key       - 一个字符串，可选值为 'calls'、'total'、'max'、'min' 或 'ave'。
      tracer_option    - 一个字符串，可选值为 'Default'、'OpDetail'、'AllOpDetail'。
      profile_path     - 一个字符串，保存序列化的性能分析数据的路径，可用于生成时间线。
      exit_on_finished - 一个布尔值。
    '''
    # 初始化方法，接受一个字符串参数作为选项设置
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
            'exit_on_finished': True,
            'timer_only': True
        }
        # 调用解析字符串方法
        self._parse_from_string(options_str)

    # 解析字符串方法，根据字符串设置选项值
    def _parse_from_string(self, options_str):
        # 遍历去除空格并按分号分隔的键值对
        for kv in options_str.replace(' ', '').split(';'):
            # 拆分键值对
            key, value = kv.split('=')
            # 根据键值对设置选项值
            if key == 'batch_range':
                # 处理列表类型的值
                value_list = value.replace('[', '').replace(']', '').split(',')
                value_list = list(map(int, value_list))
                # 检查值是否符合要求
                if len(value_list) >= 2 and value_list[0] >= 0 and value_list[1] > value_list[0]:
                    self._options[key] = value_list
            elif key == 'exit_on_finished':
                # 处理布尔类型的值
                self._options[key] = value.lower() in ("yes", "true", "t", "1")
            elif key in ['state', 'sorted_key', 'tracer_option', 'profile_path']:
                # 处理字符串类型的值
                self._options[key] = value
            elif key == 'timer_only':
                # 处理布尔类型的值
                self._options[key] = value

    # 获取选项值的方法
    def __getitem__(self, name):
        # 如果选项不存在，则抛出异常
        if self._options.get(name, None) is None:
            raise ValueError("ProfilerOptions does not have an option named %s." % name)
        # 返回选项值
        return self._options[name]
def add_profiler_step(options_str=None):
    '''
    Enable the operator-level timing using PaddlePaddle's profiler.
    The profiler uses a independent variable to count the profiler steps.
    One call of this function is treated as a profiler step.
    Args:
      profiler_options - a string to initialize the ProfilerOptions.
                         Default is None, and the profiler is disabled.
    '''
    # 如果没有传入参数 options_str，则直接返回
    if options_str is None:
        return

    # 声明全局变量
    global _prof 
    global _profiler_step_id
    global _profiler_options

    # 如果 _profiler_options 为 None，则根据 options_str 初始化 ProfilerOptions
    if _profiler_options is None:
        _profiler_options = ProfilerOptions(options_str)
    # 创建 Profiler 对象，根据配置参数进行性能分析
    if _prof is None:
        _timer_only = str(_profiler_options['timer_only']) == str(True)
        _prof = profiler.Profiler(
                   scheduler = (_profiler_options['batch_range'][0], _profiler_options['batch_range'][1]),
                   on_trace_ready = profiler.export_chrome_tracing('./profiler_log'),
                   timer_only = _timer_only)
        _prof.start()
    else:
        # 执行性能分析的下一个步骤
        _prof.step()
        
    # 如果当前步骤达到指定的 batch_range 上限，则停止性能分析
    if _profiler_step_id == _profiler_options['batch_range'][1]:
        _prof.stop()
        # 输出性能分析结果
        _prof.summary(
             op_detail=True,
             thread_sep=False,
             time_unit='ms')
        _prof = None
        # 如果设置了在完成后退出，则退出程序
        if _profiler_options['exit_on_finished']:
            sys.exit(0)

    # 增加当前步骤的计数
    _profiler_step_id += 1
```