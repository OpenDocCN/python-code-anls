# `arknights-mower\arknights_mower\utils\recruit.py`

```
#!Environment yolov8_Env
# 指定代码运行环境为yolov8_Env
# -*- coding: UTF-8 -*-
# 指定编码格式为UTF-8
"""
@Project ：arknights-mower 
@File    ：recruit.py
@Author  ：EightyDollars
@Date    ：2023/8/13 19:12
"""
# 项目信息和作者信息
from arknights_mower.utils.log import logger
# 从arknights_mower.utils.log模块中导入logger对象


def filter_result(tag_list, result_list, type=0):
    """
    temp_list
    {"tags": tag,
     "level":item['level'],
     "opers":item['opers']}
    """
    # 定义一个空列表temp_list
    temp_list = []
    # 遍历tag_list中的标签
    for tag in tag_list:
        # 记录调试信息，输出tag
        logger.debug(tag)
        # 遍历result_list中的结果字典
        for result_dict in result_list:
            # 遍历result_dict中的result列表
            for item in result_dict["result"]:
                '''高资'''
                # 如果type为0
                if type == 0:
                    # 如果tag等于result_dict中的tags并且item中的level等于result_dict中的level
                    if tag == result_dict['tags'] and item['level'] == result_dict['level']:
                        # 将item添加到temp_list中
                        temp_list.append(item)
                # 如果type为1
                elif type == 1:
                    # 如果tag等于item中的tags
                    if tag == item['tags']:
                        # 将包含tag、level和opers的字典添加到temp_list中
                        temp_list.append(
                            {"tags": tag,
                             "level": item['level'],
                             "opers": item['opers']})

    # 筛选好干员和对应tag存入返回用于后续jinja传输
    # 返回temp_list
    return temp_list
```