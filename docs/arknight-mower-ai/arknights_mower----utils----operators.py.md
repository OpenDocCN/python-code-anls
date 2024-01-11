# `arknights-mower\arknights_mower\utils\operators.py`

```
# 从 datetime 模块中导入 datetime 和 timedelta 类
from datetime import datetime, timedelta
# 从 ..data 模块中导入 agent_list
from ..data import agent_list
# 从 ..solvers.record 模块中导入 save_action_to_sqlite_decorator
from ..solvers.record import save_action_to_sqlite_decorator
# 从 ..utils.log 模块中导入 logger
from ..utils.log import logger

# 定义 Operators 类
class Operators(object):
    # 初始化类属性
    config = None
    operators = None
    exhaust_agent = []
    exhaust_group = []
    groups = None
    dorm = []
    max_resting_count = 4
    plan = None

    # 初始化方法
    def __init__(self, config, max_resting_count, plan):
        # 设置类属性值
        self.config = config
        self.operators = {}
        self.groups = {}
        self.exhaust_agent = []
        self.exhaust_group = []
        self.dorm = []
        self.max_resting_count = max_resting_count
        self.workaholic_agent = []
        self.plan = plan
        self.run_order_rooms = {}
        self.clues = []

    # 定义类的字符串表示形式
    def __repr__(self):
        return f'Operators(operators={self.operators})'

    # 定义获取当前房间的方法
    def get_current_room(self, room, bypass=False, current_index=None):
        # 从 self.operators 中获取当前房间的数据
        room_data = {v.current_index: v for k, v in self.operators.items() if v.current_room == room}
        # 从 self.plan 中获取当前房间的数据
        res = [obj['agent'] for obj in self.plan[room]]
        not_found = False
        # 遍历当前房间的数据
        for idx, op in enumerate(res):
            # 如果索引在 room_data 中，则将对应的代理人名字替换为当前房间的代理人名字
            if idx in room_data:
                res[idx] = room_data[idx].name
            else:
                res[idx] = ''
                # 如果 current_index 不为 None 且当前索引不在 current_index 中，则继续下一次循环
                if current_index is not None and idx not in current_index:
                    continue
                not_found = True
        # 如果未找到并且不绕过，则返回 None，否则返回 res
        if not_found and not bypass:
            return None
        else:
            return res
    # 预测 FIA（First Intervention Agent）是否能够执行任务
    def predict_fia(self, operators, fia_mood, hours=240):
        # 计算需要恢复的时间
        recover_hours = (24 - fia_mood) / 2
        # 遍历操作员列表，更新他们的心情值
        for agent in operators:
            agent.mood -= agent.depletion_rate * recover_hours
            # 如果心情值小于0，返回 False
            if agent.mood < 0.0:
                return False
        # 如果恢复时间大于等于指定时间或者在0到1之间，返回 True
        if recover_hours >= hours or 0 < recover_hours < 1:
            return True
        # 根据操作员的心情值排序操作员列表
        operators.sort(key=lambda x: (x.mood - x.lower_limit) / (x.upper_limit - x.lower_limit), reverse=False)
        # 更新 FIA 的心情值为最低的操作员的心情值
        fia_mood = operators[0].mood
        # 将最低心情值的操作员的心情值设为24
        operators[0].mood = 24
        # 递归调用预测函数，减去已恢复的时间
        return self.predict_fia(operators, fia_mood, hours - recover_hours)

    # 重置宿舍时间戳
    def reset_dorm_time(self):
        # 遍历操作员字典，重置宿舍操作员的时间戳
        for name in self.operators.keys():
            agent = self.operators[name]
            if agent.room.startswith("dorm"):
                agent.time_stamp = None

    # 保存动作到 SQLite 数据库的装饰器
    @save_action_to_sqlite_decorator
    # 更新操作员的详细信息，包括姓名、心情、当前房间、当前索引，还可以选择是否更新时间戳
    def update_detail(self, name, mood, current_room, current_index, update_time=False):
        # 获取指定姓名的操作员对象
        agent = self.operators[name]
        # 如果需要更新时间戳
        if update_time:
            # 如果操作员的时间戳不为空且心情值下降
            if agent.time_stamp is not None and agent.mood > mood:
                # 计算心情值下降的速率
                agent.depletion_rate = (agent.mood - mood) * 3600 / (
                    (datetime.now() - agent.time_stamp).total_seconds())
            # 更新时间戳为当前时间
            agent.time_stamp = datetime.now()
        # 如果移出宿舍，则清除对应宿舍数据 且重新记录高效组心情
        if agent.current_room.startswith('dorm') and not current_room.startswith('dorm') and agent.is_high():
            # 刷新宿舍时间
            self.refresh_dorm_time(agent.current_room, agent.current_index, {'agent': ''})
            # 如果需要更新时间戳，则更新时间戳为当前时间，否则置空
            if update_time:
                self.time_stamp = datetime.now()
            else:
                self.time_stamp = None
            # 重置心情值下降速率为0
            agent.depletion_rate = 0
        # 如果操作员的宿舍不为空且当前房间不是宿舍且操作员是高效组
        if self.get_dorm_by_name(name)[0] is not None and not current_room.startswith('dorm') and agent.is_high():
            # 获取操作员对应的宿舍对象
            _dorm = self.get_dorm_by_name(name)[1]
            # 重置宿舍名称和时间
            _dorm.name = ''
            _dorm.time = None
        # 更新操作员的当前房间、当前索引和心情值
        agent.current_room = current_room
        agent.current_index = current_index
        agent.mood = mood
        # 如果操作员当前在宿舍且是高效组且对应的宿舍时间为空，则返回当前索引
        if agent.current_room.startswith('dorm') and agent.is_high():
            for dorm in self.dorm:
                if dorm.position[0] == current_room and dorm.position[1] == current_index and dorm.time is None:
                    return current_index
        # 如果操作员的姓名是"菲亚梅塔"且时间戳为空或者时间戳早于当前时间，则返回当前索引
        if agent.name == "菲亚梅塔" and (
                self.operators["菲亚梅塔"].time_stamp is None or self.operators["菲亚梅塔"].time_stamp < datetime.now()):
            return current_index
    # 刷新宿舍时间，根据房间、位置和干员信息更新宿舍状态
    def refresh_dorm_time(self, room, index, agent):
        # 遍历宿舍列表
        for idx, dorm in enumerate(self.dorm):
            # 如果宿舍位置匹配当前房间和位置
            if dorm.position[0] == room and dorm.position[1] == index:
                # 获取干员名称
                _name = agent['agent']
                # 如果干员在高效组中并且心情状态为高
                if _name in self.operators.keys() and self.operators[_name].is_high():
                    # 记录干员名称到宿舍
                    dorm.name = _name
                    _agent = self.operators[_name]
                    # 如果干员心情状态不是满值
                    if _agent.mood != 24:
                        # 计算剩余时间并修改休息时间
                        sec_remaining = (_agent.upper_limit - _agent.mood) * (
                            (agent['time'] - _agent.time_stamp).total_seconds()) / (24 - _agent.mood)
                        dorm.time = _agent.time_stamp + timedelta(seconds=sec_remaining)
                    else:
                        # 否则直接使用当前时间
                        dorm.time = agent['time']
                else:
                    # 否则清空干员名称和时间
                    dorm.name = ''
                    dorm.time = None
                # 结束循环
                break

    # 修正宿舍状态
    def correct_dorm(self):
        # 遍历宿舍列表
        for idx, dorm in enumerate(self.dorm):
            # 如果宿舍有干员并且干员在干员字典中
            if dorm.name != "" and dorm.name in self.operators.keys():
                # 获取干员对象
                op = self.operators[dorm.name]
                # 如果宿舍位置和干员当前位置不匹配
                if not (dorm.position[0] == op.current_room and dorm.position[1] == op.current_index):
                    # 清空宿舍的干员名称和时间
                    self.dorm[idx].name = ""
                    self.dorm[idx].time = None
    # 获取刷新索引，即返回需要刷新的操作员索引列表
    def get_refresh_index(self, room, plan):
        # 初始化返回结果列表
        ret = []
        # 遍历宿舍列表
        for idx, dorm in enumerate(self.dorm):
            # 如果索引超过最大休息数量，则跳出循环
            if idx >= self.max_resting_count:
                break
            # 如果宿舍位置与指定房间相同
            if dorm.position[0] == room:
                # 遍历计划列表
                for i, _name in enumerate(plan):
                    # 如果计划中的操作员在操作员字典中，并且优先级高，并且休息优先级为高，并且不是以"dorm"开头的房间
                    if _name in self.operators.keys() and self.operators[_name].is_high() and self.operators[
                        _name].resting_priority == 'high' and not self.operators[_name].room.startswith('dorm'):
                        # 将索引添加到返回结果列表中
                        ret.append(i)
                # 跳出循环
                break
        # 返回结果列表
        return ret

    # 根据名称获取宿舍
    def get_dorm_by_name(self, name):
        # 遍历宿舍列表
        for idx, dorm in enumerate(self.dorm):
            # 如果宿舍名称与指定名称相同
            if dorm.name == name:
                # 返回索引和宿舍对象
                return idx, dorm
        # 如果没有找到对应名称的宿舍，则返回None
        return None, None
    # 定义一个方法，用于向系统中添加新的操作员
    def add(self, operator):
        # 如果操作员不在操作员列表中，则直接返回
        if operator.name not in agent_list:
            return
        # 如果操作员在配置文件中，并且配置文件中包含'休息优先级'，则将操作员的休息优先级设置为配置文件中的值
        if operator.name in self.config.keys() and 'RestingPriority' in self.config[operator.name].keys():
            operator.resting_priority = self.config[operator.name]['RestingPriority']
        # 如果操作员在配置文件中，并且配置文件中包含'耗尽需求'，则将操作员的耗尽需求设置为配置文件中的值
        if operator.name in self.config.keys() and 'ExhaustRequire' in self.config[operator.name].keys():
            operator.exhaust_require = self.config[operator.name]['ExhaustRequire']
        # 如果操作员在配置文件中，并且配置文件中包含'满状态休息'，则将操作员的满状态休息设置为配置文件中的值
        if operator.name in self.config.keys() and 'RestInFull' in self.config[operator.name].keys():
            operator.rest_in_full = self.config[operator.name]['RestInFull']
        # 如果操作员在配置文件中，并且配置文件中包含'下限'，则将操作员的下限设置为配置文件中的值
        if operator.name in self.config.keys() and 'LowerLimit' in self.config[operator.name].keys():
            operator.lower_limit = self.config[operator.name]['LowerLimit']
        # 如果操作员在配置文件中，并且配置文件中包含'上限'，则将操作员的上限设置为配置文件中的值
        if operator.name in self.config.keys() and 'UpperLimit' in self.config[operator.name].keys():
            operator.upper_limit = self.config[operator.name]['UpperLimit']
        # 如果操作员在配置文件中，并且配置文件中包含'工作狂'，则将操作员的工作狂属性设置为配置文件中的值
        if operator.name in self.config.keys() and 'Workaholic' in self.config[operator.name].keys():
            operator.workaholic = self.config[operator.name]['Workaholic']
        # 将操作员添加到操作员列表中
        self.operators[operator.name] = operator
        # 如果需要用尽心情的操作员或者所属组在用尽组中，并且操作员不在用尽操作员列表中，则将操作员添加到用尽操作员列表中
        if (operator.exhaust_require or operator.group in self.exhaust_group) \
                and operator.name not in self.exhaust_agent:
            self.exhaust_agent.append(operator.name)
            # 如果操作员所属组不为空，则将其所属组添加到用尽组列表中
            if operator.group != '':
                self.exhaust_group.append(operator.group)
        # 如果操作员所属组不为空，则将操作员添加到对应的组中
        if operator.group != "":
            if operator.group not in self.groups.keys():
                self.groups[operator.group] = [operator.name]
            else:
                self.groups[operator.group].append(operator.name)
        # 如果操作员是工作狂，并且不在工作狂操作员列表中，则将其添加到工作狂操作员列表中
        if operator.workaholic and operator.name not in self.workaholic_agent:
            self.workaholic_agent.append(operator.name)
    # 返回可用的空闲房间数量
    def available_free(self, free_type='high'):
        # 初始化返回值
        ret = 0
        # 如果空闲类型为'high'
        if free_type == 'high':
            # 初始化索引
            idx = 0
            # 遍历宿舍列表
            for dorm in self.dorm:
                # 如果房间名为空，或者房间名在操作员字典中且不是高优先级
                if dorm.name == '' or (dorm.name in self.operators.keys() and not self.operators[dorm.name].is_high()):
                    # 空闲房间数量加一
                    ret += 1
                # 如果房间有休息时间且时间早于当前时间
                elif dorm.time is not None and dorm.time < datetime.now():
                    # 记录日志
                    logger.info("检测到房间休息完毕，释放Free位")
                    # 重置房间名为空
                    dorm.name = ''
                    # 空闲房间数量加一
                    ret += 1
                # 如果索引达到最大休息数量减一
                if idx == self.max_resting_count - 1:
                    # 退出循环
                    break
                else:
                    # 索引加一
                    idx += 1
        # 如果空闲类型不为'high'
        else:
            # 初始化索引为最大休息数量
            idx = self.max_resting_count
            # 遍历宿舍列表
            for i in range(idx, len(self.dorm)):
                # 获取当前房间
                dorm = self.dorm[i]
                # 如果房间名为空，或者房间名在操作员字典中且不是高优先级
                # TODO 高效组且低优先可以相互替换
                if dorm.name == '' or (dorm.name in self.operators.keys() and not self.operators[dorm.name].is_high()):
                    # 重置房间名为空
                    dorm.name = ''
                    # 空闲房间数量加一
                    ret += 1
                # 如果房间有休息时间且时间早于当前时间
                elif dorm.time is not None and dorm.time < datetime.now():
                    # 记录日志
                    logger.info("检测到房间休息完毕，释放Free位")
                    # 重置房间名为空
                    dorm.name = ''
                    # 空闲房间数量加一
                    ret += 1
        # 返回空闲房间数量
        return ret

    # 分配宿舍给操作员
    def assign_dorm(self, name):
        # 判断操作员的休息优先级是否为高
        is_high = self.operators[name].resting_priority == 'high'
        # 如果是高优先级
        if is_high:
            # 查找第一个空闲房间或者非高优先级的房间
            _room = next(obj for obj in self.dorm if
                         obj.name not in self.operators.keys() or not self.operators[obj.name].is_high())
        # 如果不是高优先级
        else:
            # 初始化房间为空
            _room = None
            # 遍历宿舍列表
            for i in range(self.max_resting_count, len(self.dorm)):
                # 如果房间名为空
                if self.dorm[i].name == '':
                    # 分配房间给操作员
                    _room = self.dorm[i]
                    # 退出循环
                    break
        # 设置房间名为操作员名
        _room.name = name
        # 返回分配的房间
        return _room
    # 定义一个打印方法，返回一个表示对象内容的字符串
    def print(self):
        # 初始化返回字符串
        ret = "{"
        # 初始化操作符列表和宿舍列表
        op = []
        dorm = []
        # 遍历操作符字典，将操作符和其属性转换成字符串并添加到操作符列表中
        for k, v in self.operators.items():
            op.append("'" + k + "': " + str(vars(v)))
        # 将操作符列表转换成字符串，添加到返回字符串中
        ret += "'operators': {" + ','.join(op) + "},"
        # 遍历宿舍列表，将宿舍属性转换成字符串并添加到宿舍列表中
        for v in self.dorm:
            dorm.append(str(vars(v)))
        # 将宿舍列表转换成字符串，添加到返回字符串中
        ret += "'dorms': [" + ','.join(dorm) + "]}"
        # 返回表示对象内容的字符串
        return ret
# 定义 Dormitory 类
class Dormitory(object):

    # 初始化方法，设置宿舍的位置、名称和时间
    def __init__(self, position, name='', time=None):
        self.position = position
        self.name = name
        self.time = time

    # 返回对象的字符串表示形式
    def __repr__(self):
        return f"Dormitory(position={self.position},name='{self.name}',time='{self.time}')"


# 定义 Operator 类
class Operator(object):

    # 类属性
    time_stamp = None
    depletion_rate = 0
    workaholic = False

    # 初始化方法，设置操作员的各种属性
    def __init__(self, name, room, index=-1, group='', replacement=[], resting_priority='low', current_room='',
                 exhaust_require=False,
                 mood=24, upper_limit=24, rest_in_full=False, current_index=-1, lower_limit=0, operator_type="low",
                 depletion_rate=0, time_stamp=None):
        self.name = name
        self.room = room
        self.operator_type = operator_type
        self.index = index
        self.group = group
        self.replacement = replacement
        self.resting_priority = resting_priority
        self.current_room = current_room
        self.exhaust_require = exhaust_require
        self.upper_limit = upper_limit
        self.rest_in_full = rest_in_full
        self.mood = mood
        self.current_index = current_index
        self.lower_limit = lower_limit
        self.depletion_rate = depletion_rate
        self.time_stamp = time_stamp

    # 判断操作员类型是否为高级
    def is_high(self):
        return self.operator_type == 'high'

    # 判断是否需要刷新心情
    def need_to_refresh(self, h=2, r=""):
        # 是否需要读取心情
        if self.operator_type == 'high':
            if self.time_stamp is None or (
                    self.time_stamp is not None and self.time_stamp + timedelta(hours=h) < datetime.now()) or (
                    r.startswith("dorm") and not self.room.startswith("dorm")):
                return True
        return False
    # 检查是否为有效操作员
    def not_valid(self):
        # 如果是工作狂，则无效
        if self.workaholic:
            return False
        # 如果操作员类型为高级
        if self.operator_type == 'high':
            # 如果当前房间不以"dorm"开头且目标房间以"dorm"开头，并且心情为-1或24，则有效
            if not self.room.startswith("dorm") and self.current_room.startswith("dorm"):
                if self.mood == -1 or self.mood == 24:
                    return True
                else:
                    return False
            # 否则，需要刷新或者当前房间与目标房间不同，或者索引不同，则有效
            return self.need_to_refresh(2.5) or self.current_room != self.room or self.index != self.current_index
        # 其他情况均为无效
        return False

    # 获取当前心情值
    def current_mood(self):
        # 预测心情值为当前心情值
        predict = self.mood
        # 如果有时间戳
        if self.time_stamp is not None:
            # 根据时间戳和心情值递减速率计算预测心情值
            predict = self.mood - self.depletion_rate * (datetime.now() - self.time_stamp).total_seconds() / 3600
        # 如果预测心情值在0到24之间，则返回预测心情值，否则返回当前心情值
        if 0 <= predict <= 24:
            return predict
        else:
            return self.mood

    # 返回操作员对象的字符串表示形式
    def __repr__(self):
        return f"Operator(name='{self.name}', room='{self.room}', index={self.index}, group='{self.group}', replacement={self.replacement}, resting_priority='{self.resting_priority}', current_room='{self.current_room}',exhaust_require={self.exhaust_require},mood={self.mood}, upper_limit={self.upper_limit}, rest_in_full={self.rest_in_full}, current_index={self.current_index}, lower_limit={self.lower_limit}, operator_type='{self.operator_type}',depletion_rate={self.depletion_rate},time_stamp='{self.time_stamp}')"
```