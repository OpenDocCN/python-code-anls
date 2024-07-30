# `.\yolov8\tests\test_explorer.py`

```py
# 导入必要的库和模块：PIL 图像处理库和 pytest 测试框架
import PIL
import pytest

# 从 ultralytics 包中导入 Explorer 类和 ASSETS 资源
from ultralytics import Explorer
from ultralytics.utils import ASSETS

# 使用 pytest 的标记 @pytest.mark.slow 标记此函数为慢速测试
@pytest.mark.slow
def test_similarity():
    """测试 Explorer 中相似性计算和 SQL 查询的正确性和返回长度。"""
    # 创建 Explorer 对象，使用配置文件 'coco8.yaml'
    exp = Explorer(data="coco8.yaml")
    # 创建嵌入表格
    exp.create_embeddings_table()
    # 获取索引为 1 的相似项
    similar = exp.get_similar(idx=1)
    # 断言相似项的长度为 4
    assert len(similar) == 4
    # 使用图像文件 'bus.jpg' 获取相似项
    similar = exp.get_similar(img=ASSETS / "bus.jpg")
    # 断言相似项的长度为 4
    assert len(similar) == 4
    # 获取索引为 [1, 2] 的相似项，限制返回结果为 2 个
    similar = exp.get_similar(idx=[1, 2], limit=2)
    # 断言相似项的长度为 2
    assert len(similar) == 2
    # 获取相似性索引
    sim_idx = exp.similarity_index()
    # 断言相似性索引的长度为 4
    assert len(sim_idx) == 4
    # 执行 SQL 查询，查询条件为 'labels LIKE '%zebra%''
    sql = exp.sql_query("WHERE labels LIKE '%zebra%'")
    # 断言 SQL 查询结果的长度为 1
    assert len(sql) == 1


@pytest.mark.slow
def test_det():
    """测试检测功能，并验证嵌入表格是否包含边界框。"""
    # 创建 Explorer 对象，使用配置文件 'coco8.yaml' 和模型 'yolov8n.pt'
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    # 强制创建嵌入表格
    exp.create_embeddings_table(force=True)
    # 断言表格中的边界框列的长度大于 0
    assert len(exp.table.head()["bboxes"]) > 0
    # 获取索引为 [1, 2] 的相似项，限制返回结果为 10 个
    similar = exp.get_similar(idx=[1, 2], limit=10)
    # 断言相似项的长度大于 0
    assert len(similar) > 0
    # 执行绘制相似项的操作，返回值应为 PIL 图像对象
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    # 断言返回值是 PIL 图像对象
    assert isinstance(similar, PIL.Image.Image)


@pytest.mark.slow
def test_seg():
    """测试分割功能，并确保嵌入表格包含分割掩码。"""
    # 创建 Explorer 对象，使用配置文件 'coco8-seg.yaml' 和模型 'yolov8n-seg.pt'
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    # 强制创建嵌入表格
    exp.create_embeddings_table(force=True)
    # 断言表格中的分割掩码列的长度大于 0
    assert len(exp.table.head()["masks"]) > 0
    # 获取索引为 [1, 2] 的相似项，限制返回结果为 10 个
    similar = exp.get_similar(idx=[1, 2], limit=10)
    # 断言相似项的长度大于 0
    assert len(similar) > 0
    # 执行绘制相似项的操作，返回值应为 PIL 图像对象
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    # 断言返回值是 PIL 图像对象
    assert isinstance(similar, PIL.Image.Image)


@pytest.mark.slow
def test_pose():
    """测试姿势估计功能，并验证嵌入表格是否包含关键点。"""
    # 创建 Explorer 对象，使用配置文件 'coco8-pose.yaml' 和模型 'yolov8n-pose.pt'
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    # 强制创建嵌入表格
    exp.create_embeddings_table(force=True)
    # 断言表格中的关键点列的长度大于 0
    assert len(exp.table.head()["keypoints"]) > 0
    # 获取索引为 [1, 2] 的相似项，限制返回结果为 10 个
    similar = exp.get_similar(idx=[1, 2], limit=10)
    # 断言相似项的长度大于 0
    assert len(similar) > 0
    # 执行绘制相似项的操作，返回值应为 PIL 图像对象
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    # 断言返回值是 PIL 图像对象
    assert isinstance(similar, PIL.Image.Image)
```