# `arknights-mower\arknights_mower\utils\scene.py`

```
# 从上级目录的 data 模块中导入 scene_list 数据
from ..data import scene_list

# 定义一个名为 Scene 的类
class Scene:
    pass

# 定义一个空的字典 SceneComment
SceneComment = {}

# 遍历 scene_list 中的每个场景
for scene in scene_list.keys():
    # 将场景名转换为整数类型
    id = int(scene)
    # 获取场景对应的标签
    label = scene_list[scene]['label']
    # 获取场景对应的注释
    comment = scene_list[scene]['comment']
    # 将标签和对应的整数值设置为 Scene 类的属性
    setattr(Scene, label, id)
    # 将场景的整数值和对应的注释添加到 SceneComment 字典中
    SceneComment[id] = comment
```