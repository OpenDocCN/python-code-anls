# `yolov5-DNF\datasets_utils.py`

```py
# 导入 os 模块
import os
# 导入 shutil 模块
import shutil

# 设置原始路径和目标路径
root_path = "datasets/guiqi/patch1"
yolo5_data_dir = "datasets/guiqi/patch1_yolo5"

# 初始化存储文件名的列表
json_list = []
imgs_list = []

# 获取原始路径下的所有文件和文件夹
dir = os.listdir(root_path)

# 遍历原始路径下的所有文件和文件夹
for d in dir:
    # 如果文件名以 .json 结尾，则将其对应的图片文件名和标签文件名分别添加到对应的列表中
    if d.endswith(".json"):
        imgs_list.append(d.strip().split(".")[0] + ".jpg")
        json_list.append(d)

# 打印图片文件名列表和标签文件名列表
print(imgs_list)
print(json_list)

# 遍历图片文件名列表和标签文件名列表，将它们分别复制到目标路径下的 imgs 和 labels_json 文件夹中
for img_name, json in zip(imgs_list, json_list):
    shutil.copy(os.path.join(root_path + "/" + img_name), os.path.join(yolo5_data_dir + '/imgs'))
    shutil.copy(os.path.join(root_path + "/" + json), os.path.join(yolo5_data_dir + '/labels_json'))

# 以下代码段被注释掉，不会执行
# # 选一部分数据作为验证集
# img_train_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\train\images"
# img_valid_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\valid\images"
# label_train_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\train\labels"
# label_valid_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\valid\labels"
# eval_ratio = 0.1
# dir = os.listdir(img_train_path)
# eval_nums = int(eval_ratio * len(dir))
# import random
# random.shuffle(dir)
# for d in dir[:eval_nums]:
#     shutil.move(os.path.join(img_train_path + "\\" + d), os.path.join(img_valid_path + "\\" + d))
#     shutil.move(os.path.join(label_train_path + "\\" + d.strip().split(".")[0] + ".txt"),
#                 os.path.join(label_valid_path + "\\" + d.strip().split(".")[0] + ".txt"))

# 以下代码段被注释掉，不会执行
# undict生成
#
# name2id = {'hero': 0, 'small_map': 1, "monster": 2, 'money': 3, 'material': 4, 'door': 5, 'BOSS': 6, 'box': 7, 'options': 8}
# id2name = {}
# for key, val in name2id.items():
#     id2name[val] = key
# print(id2name)
```