# `arknights-mower\arknights_mower\solvers\record.py`

```py
# 导入sqlite3模块，用于操作SQLite数据库
import sqlite3
# 导入os模块，用于提供与操作系统交互的功能
import os
# 导入defaultdict类，用于创建一个字典，当字典中的key不存在时，返回一个默认值
from collections import defaultdict
# 从arknights_mower.utils.log模块中导入logger对象
from arknights_mower.utils.log import logger
# 从datetime模块中导入datetime和timezone类
from datetime import datetime, timezone

# 定义一个装饰器函数，用于记录干员进出站以及心情数据，并将记录信息存入agent_action表里
def save_action_to_sqlite_decorator(func):
    # 定义一个装饰器函数，接受干员名字、心情、当前房间、当前索引和更新时间作为参数
    def wrapper(self, name, mood, current_room, current_index, update_time=False):
        agent = self.operators[name]  # 获取干员对象

        agent_current_room = agent.current_room  # 获取干员当前所在房间
        agent_is_high = agent.is_high()  # 判断干员是否高优先级

        # 调用原函数
        result = func(self, name, mood, current_room, current_index, update_time)
        if not update_time:
            return
        # 如果不需要更新时间，则直接返回

        # 保存到数据库
        current_time = datetime.now()  # 获取当前时间
        database_path = os.path.join('tmp', 'data.db')  # 数据库文件路径

        try:
            # 如果不存在'tmp'目录，则创建
            os.makedirs('tmp', exist_ok=True)

            connection = sqlite3.connect(database_path)  # 连接数据库
            cursor = connection.cursor()  # 创建游标

            # 如果表不存在，则创建表
            cursor.execute('CREATE TABLE IF NOT EXISTS agent_action ('
                           'name TEXT,'
                           'agent_current_room TEXT,'
                           'current_room TEXT,'
                           'is_high INTEGER,'
                           'agent_group TEXT,'
                           'mood REAL,'
                           'current_time TEXT'
                           ')')

            # 插入数据
            cursor.execute('INSERT INTO agent_action VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (name, agent_current_room, current_room, int(agent_is_high), agent.group, mood,
                            str(current_time)))

            connection.commit()  # 提交事务
            connection.close()  # 关闭连接

            # 记录操作
            logger.debug(
                f"Saved action to SQLite: Name: {name}, Agent's Room: {agent_current_room}, Agent's group: {agent.group}, "
                f"Current Room: {current_room}, Is High: {agent_is_high}, Current Time: {current_time}")

        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")  # 记录错误信息

        return result  # 返回结果

    return wrapper  # 返回装饰器函数
def get_work_rest_ratios():
    # TODO 整理数据计算工休比
    # 设置数据库路径
    database_path = os.path.join('tmp', 'data.db')

    try:
        # 连接到数据库
        conn = sqlite3.connect(database_path)
        # 创建游标对象
        cursor = conn.cursor()
        # 查询数据
        cursor.execute("""
                        SELECT a.*
                        FROM agent_action a
                        JOIN (
                            SELECT DISTINCT b.name
                            FROM agent_action b
                            WHERE DATE(b.current_time) >= DATE('now', '-7 day', 'localtime')
                            AND b.is_high = 1 AND b.current_room NOT LIKE 'dormitory%'
                            UNION
                            SELECT '菲亚梅塔' AS name
                        ) AS subquery ON a.name = subquery.name
                        WHERE DATE(a.current_time) >= DATE('now', '-1 month', 'localtime')
                        ORDER BY a.current_time;
                       """)
        # 获取查询结果
        data = cursor.fetchall()
        # 关闭数据库连接
        conn.close()
    except sqlite3.Error as e:
        # 如果发生异常，将数据置为空列表
        data = []
    # 初始化处理后的数据和分组数据
    processed_data = {}
    grouped_data = {}
    # 遍历查询结果
    for row in data:
        # 获取行数据中的姓名、当前房间和当前时间
        name = row[0]
        current_room = row[2]
        current_time = row[6]  # Assuming index 6 is the current_time column
        # 获取分组数据中对应姓名的代理对象，如果不存在则创建
        agent = grouped_data.get(name, {
            'agent_data': [{'current_time': current_time,
                            'current_room': current_room}],
            'difference': []
        })
        # 计算时间差并添加到代理对象的差异列表中
        difference = {'time_diff': calculate_time_difference(agent['agent_data'][-1]['current_time'], current_time),
                      'current_room': agent['agent_data'][-1]['current_room']}
        agent['agent_data'].append({'current_time': current_time,
                                    'current_room': current_room})
        agent['difference'].append(difference)
        # 更新分组数据中对应姓名的代理对象
        grouped_data[name] = agent
    # 遍历分组后的数据
    for name in grouped_data:
        # 初始化工作时间和休息时间
        work_time = 0
        rest_time = 0
        # 遍历每个人的时间差数据
        for difference in grouped_data[name]['difference']:
            # 如果当前房间以'dormitory'开头，将时间差加到休息时间上
            if difference['current_room'].startswith('dormitory'):
                rest_time += difference['time_diff']
            # 否则将时间差加到工作时间上
            else:
                work_time += difference['time_diff']
        # 将处理后的数据存入processed_data字典中
        processed_data[name] = {'labels':['休息时间','工作时间'],
                                'datasets':[{
                                    'data':[rest_time,work_time]
                                }]}
    # 返回处理后的数据
    return processed_data
# 整理心情曲线
def get_mood_ratios():
    # 数据库文件路径
    database_path = os.path.join('tmp', 'data.db')

    try:
        # 连接到数据库
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        # 查询数据（筛掉宿管和替班组的数据）
        cursor.execute("""
                       SELECT a.*
                        FROM agent_action a
                        JOIN (
                            SELECT DISTINCT b.name
                            FROM agent_action b
                            WHERE DATE(b.current_time) >= DATE('now', '-7 day', 'localtime')
                            AND b.is_high = 1 AND b.current_room NOT LIKE 'dormitory%'
                            UNION
                            SELECT '菲亚梅塔' AS name
                        ) AS subquery ON a.name = subquery.name
                        WHERE DATE(a.current_time) >= DATE('now', '-7 day', 'localtime')
                        ORDER BY a.agent_group DESC, a.current_time;

        """)
        # 获取查询结果
        data = cursor.fetchall()
        # 关闭数据库连接
        conn.close()
    except sqlite3.Error as e:
        # 出现异常时，将数据置为空列表
        data = []

    # 获取工作和休息数据的比例
    work_rest_data_ratios = get_work_rest_ratios()
    # 初始化分组数据字典
    grouped_data = {}
    # 初始化分组工作和休息数据字典
    grouped_work_rest_data = {}
    # 遍历数据中的每一行
    for row in data:
        # 获取组名，假设'agent_group'在索引4处
        group_name = row[4]  
        # 如果组名为空，则使用第一列的值作为组名
        if not group_name:
            group_name = row[0]
        # 获取或创建以组名为键的心情数据
        mood_data = grouped_data.get(group_name, {
            'labels': [],
            'datasets': []
        })
        # 获取或创建以组名为键的工作休息数据
        work_rest_data = grouped_work_rest_data.get(group_name,
            work_rest_data_ratios[row[0]]
        )
        # 将工作休息数据存入以组名为键的字典中
        grouped_work_rest_data[group_name]=work_rest_data

        # 将时间戳字符串转换为 Luxon 格式的字符串
        timestamp_datetime = datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S.%f')  # 假设'current_time'在索引6处
        current_time = f"{timestamp_datetime.year:04d}-{timestamp_datetime.month:02d}-{timestamp_datetime.day:02d}T{timestamp_datetime.hour:02d}:{timestamp_datetime.minute:02d}:{timestamp_datetime.second:02d}.{timestamp_datetime.microsecond:06d}+08:00"

        # 获取心情标签，假设'name'在索引0处
        mood_label = row[0]  
        # 获取心情数值，假设'mood'在索引5处
        mood_value = row[5]  

        # 如果心情标签已存在于数据集中，则将当前时间和数值添加到对应的数据集中
        if mood_label in [dataset['label'] for dataset in mood_data['datasets']]:
            mood_data['labels'].append(current_time)
            for dataset in mood_data['datasets']:
                if dataset['label'] == mood_label:
                    dataset['data'].append({'x': current_time, 'y': mood_value})
                    break
        else:
            # 如果心情标签不存在于数据集中，则创建一个新的数据集
            mood_data['labels'].append(current_time)
            mood_data['datasets'].append({
                'label': mood_label,
                'data': [{'x': current_time, 'y': mood_value}]
            })

        # 将以组名为键的心情数据存入字典中
        grouped_data[group_name] = mood_data
    # 打印以组名为键的工作休息数据
    print(grouped_work_rest_data)
    # 初始化格式化后的数据数组
    formatted_data = []
    # 遍历分组数据的字典，获取每个组名和对应的心情数据
    for group_name, mood_data in grouped_data.items():
        # 将格式化后的数据添加到列表中
        formatted_data.append({
            'groupName': group_name,  # 将组名添加到格式化数据中
            'moodData': mood_data,  # 将心情数据添加到格式化数据中
            'workRestData':grouped_work_rest_data[group_name]  # 将对应组名的工作休息数据添加到格式化数据中
        })
    # 返回格式化后的数据
    return formatted_data
# 计算两个时间之间的时间差
def calculate_time_difference(start_time, end_time):
    # 定义时间格式
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    # 将开始时间字符串转换为 datetime 对象
    start_datetime = datetime.strptime(start_time, time_format)
    # 将结束时间字符串转换为 datetime 对象
    end_datetime = datetime.strptime(end_time, time_format)
    # 计算时间差
    time_difference = end_datetime - start_datetime
    # 返回时间差的秒数
    return time_difference.total_seconds()

# 主函数，调用获取工作和休息比例的函数
def __main__():
    get_work_rest_ratios()

# 调用主函数
__main__()
```