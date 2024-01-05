# `.\Zelda-with-Python\Code\Support.py`

```
# 从 csv 模块中导入 reader 函数
from csv import reader
# 导入 os 模块
import os
# 从 os 模块中导入 walk 函数
from os import walk
# 导入 pygame 模块

# 用于将 CSV 文件导入到 Python 中以及其他一些功能

# 这是用于文件（特别是图片）导入的（这一行将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def import_csv_layout(path): 
    # 创建一个空的地形地图列表
    terrain_map = []

    # 打开指定路径的文件
    with open(path) as level_map:
        # 使用 csv 模块的 reader 函数读取文件内容，以逗号为分隔符
        layout = reader(level_map, delimiter = ",")

        # 遍历文件内容的每一行
        for row in layout:
            # 将每一行转换为列表，并添加到地形地图列表中
            terrain_map.append(list(row))
# 返回地形地图
def import_folder(path):
    # 创建一个空列表来存储图像表面
    surface_list = []
    
    # 遍历指定路径下的所有文件
    for _, __, img_files in walk(path):
        # 遍历文件夹中的所有图像文件
        for image in img_files:
            # 获取图像文件的完整路径
            full_path = path + "/" + image
            # 加载图像并转换为带透明度的表面
            image_surf = pygame.image.load(full_path).convert_alpha()
            # 将图像表面添加到列表中
            surface_list.append(image_surf)
    # 返回图像表面列表
    return surface_list
```