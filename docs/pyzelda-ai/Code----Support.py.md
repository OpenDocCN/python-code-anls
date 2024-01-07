# `.\Zelda-with-Python\Code\Support.py`

```

# 从csv模块中导入reader函数
from csv import reader
# 导入os模块
import os
# 从os模块中导入walk函数
from os import walk
# 导入pygame模块

# 用于将CSV文件导入Python以及其他相关操作

# 这是用于文件（特别是图片）导入的代码（这行将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def import_csv_layout(path): 
    # 创建一个空的地形地图列表
    terrain_map = []

    # 打开指定路径的文件
    with open(path) as level_map:
        # 使用CSV reader函数读取文件，以逗号为分隔符
        layout = reader(level_map, delimiter = ",")

        # 遍历CSV文件的每一行
        for row in layout:
            # 将每一行转换为列表，并添加到地形地图列表中
            terrain_map.append(list(row))
        
        # 返回地形地图列表
        return terrain_map


def import_folder(path):
    # 创建一个空的表面列表
    surface_list = []
    
    # 遍历指定路径下的所有文件和子目录
    for _, __, img_files in walk(path):
        # 遍历每个图片文件
        for image in img_files:
            # 拼接完整的文件路径
            full_path = path + "/" + image
            # 加载图片并转换为alpha通道
            image_surf = pygame.image.load(full_path).convert_alpha()
            # 将图片表面添加到表面列表中
            surface_list.append(image_surf)
    # 返回表面列表
    return surface_list

```