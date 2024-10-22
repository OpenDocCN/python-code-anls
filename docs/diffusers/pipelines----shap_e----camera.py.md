# `.\diffusers\pipelines\shap_e\camera.py`

```py
# Copyright 2024 Open AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数据类装饰器，用于简化类的定义
from dataclasses import dataclass
# 导入元组类型，用于类型注解
from typing import Tuple

# 导入 NumPy 库，进行数值计算
import numpy as np
# 导入 PyTorch 库，进行张量操作
import torch


# 定义一个可微分的投影相机类
@dataclass
class DifferentiableProjectiveCamera:
    """
    Implements a batch, differentiable, standard pinhole camera
    """

    # 相机的原点，形状为 [batch_size x 3]
    origin: torch.Tensor  # [batch_size x 3]
    # x 轴方向向量，形状为 [batch_size x 3]
    x: torch.Tensor  # [batch_size x 3]
    # y 轴方向向量，形状为 [batch_size x 3]
    y: torch.Tensor  # [batch_size x 3]
    # z 轴方向向量，形状为 [batch_size x 3]
    z: torch.Tensor  # [batch_size x 3]
    # 相机的宽度
    width: int
    # 相机的高度
    height: int
    # 水平视场角
    x_fov: float
    # 垂直视场角
    y_fov: float
    # 相机的形状信息，元组类型
    shape: Tuple[int]

    # 初始化后进行验证
    def __post_init__(self):
        # 验证原点和方向向量的批次大小一致
        assert self.x.shape[0] == self.y.shape[0] == self.z.shape[0] == self.origin.shape[0]
        # 验证每个方向向量的维度为3
        assert self.x.shape[1] == self.y.shape[1] == self.z.shape[1] == self.origin.shape[1] == 3
        # 验证每个张量的维度都是2
        assert len(self.x.shape) == len(self.y.shape) == len(self.z.shape) == len(self.origin.shape) == 2

    # 返回相机的分辨率
    def resolution(self):
        # 将宽度和高度转为浮点型张量
        return torch.from_numpy(np.array([self.width, self.height], dtype=np.float32))

    # 返回相机的视场角
    def fov(self):
        # 将水平和垂直视场角转为浮点型张量
        return torch.from_numpy(np.array([self.x_fov, self.y_fov], dtype=np.float32))

    # 获取图像坐标
    def get_image_coords(self) -> torch.Tensor:
        """
        :return: coords of shape (width * height, 2)
        """
        # 生成像素索引，范围从 0 到 width * height - 1
        pixel_indices = torch.arange(self.height * self.width)
        # 计算坐标，分离出 x 和 y 组件
        coords = torch.stack(
            [
                pixel_indices % self.width,  # x 坐标
                torch.div(pixel_indices, self.width, rounding_mode="trunc"),  # y 坐标
            ],
            axis=1,  # 沿新轴堆叠
        )
        # 返回坐标张量
        return coords

    # 计算相机光线
    @property
    def camera_rays(self):
        # 获取批次大小和其他形状信息
        batch_size, *inner_shape = self.shape
        # 计算内部批次大小
        inner_batch_size = int(np.prod(inner_shape))

        # 获取图像坐标
        coords = self.get_image_coords()
        # 将坐标广播到批次大小
        coords = torch.broadcast_to(coords.unsqueeze(0), [batch_size * inner_batch_size, *coords.shape])
        # 获取相机光线
        rays = self.get_camera_rays(coords)

        # 调整光线张量的形状
        rays = rays.view(batch_size, inner_batch_size * self.height * self.width, 2, 3)

        # 返回光线张量
        return rays
    # 获取相机射线的函数，输入为坐标张量，输出为射线张量
    def get_camera_rays(self, coords: torch.Tensor) -> torch.Tensor:
        # 提取批大小、形状和坐标数量
        batch_size, *shape, n_coords = coords.shape
        # 确保坐标数量为 2
        assert n_coords == 2
        # 确保批大小与原点的数量一致
        assert batch_size == self.origin.shape[0]

        # 将坐标展平，形状变为 (batch_size, -1, 2)
        flat = coords.view(batch_size, -1, 2)

        # 获取分辨率
        res = self.resolution()
        # 获取视场角
        fov = self.fov()

        # 计算归一化坐标，范围从 -1 到 1
        fracs = (flat.float() / (res - 1)) * 2 - 1
        # 将归一化坐标转换为视场角下的方向
        fracs = fracs * torch.tan(fov / 2)

        # 将归一化坐标重新调整形状
        fracs = fracs.view(batch_size, -1, 2)
        # 计算射线方向
        directions = (
            self.z.view(batch_size, 1, 3)  # z 方向
            + self.x.view(batch_size, 1, 3) * fracs[:, :, :1]  # x 方向
            + self.y.view(batch_size, 1, 3) * fracs[:, :, 1:]  # y 方向
        )
        # 对方向进行归一化
        directions = directions / directions.norm(dim=-1, keepdim=True)
        # 堆叠原点和方向形成射线张量
        rays = torch.stack(
            [
                # 扩展原点以匹配方向的形状
                torch.broadcast_to(self.origin.view(batch_size, 1, 3), [batch_size, directions.shape[1], 3]),
                directions,  # 射线方向
            ],
            dim=2,  # 在最后一个维度进行堆叠
        )
        # 返回最终的射线张量，形状为 (batch_size, *shape, 2, 3)
        return rays.view(batch_size, *shape, 2, 3)

    # 调整图像大小的函数，返回新的相机对象
    def resize_image(self, width: int, height: int) -> "DifferentiableProjectiveCamera":
        """
        创建一个新的相机用于调整后的视图，假设长宽比不变。
        """
        # 确保宽高比不变
        assert width * self.height == height * self.width, "The aspect ratio should not change."
        # 返回新的可微分投影相机对象
        return DifferentiableProjectiveCamera(
            origin=self.origin,  # 原点
            x=self.x,  # x 方向
            y=self.y,  # y 方向
            z=self.z,  # z 方向
            width=width,  # 新的宽度
            height=height,  # 新的高度
            x_fov=self.x_fov,  # x 方向的视场角
            y_fov=self.y_fov,  # y 方向的视场角
        )
# 创建一个全景摄像机的函数，返回一个可微分的投影摄像机
def create_pan_cameras(size: int) -> DifferentiableProjectiveCamera:
    # 初始化原点、x轴、y轴和z轴的列表
    origins = []
    xs = []
    ys = []
    zs = []
    # 生成20个从0到2π的均匀分布的角度
    for theta in np.linspace(0, 2 * np.pi, num=20):
        # 计算z轴方向的单位向量
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        # 将z向量标准化
        z /= np.sqrt(np.sum(z**2))
        # 计算相机原点位置，向外移动4个单位
        origin = -z * 4
        # 计算x轴方向的向量
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        # 计算y轴方向的向量，通过z和x的叉积获得
        y = np.cross(z, x)
        # 将计算得到的原点和轴向向量添加到对应的列表中
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    # 返回一个DifferentiableProjectiveCamera对象，包含原点、轴向向量和其它参数
    return DifferentiableProjectiveCamera(
        # 将原点列表转换为PyTorch的张量
        origin=torch.from_numpy(np.stack(origins, axis=0)).float(),
        # 将x轴列表转换为PyTorch的张量
        x=torch.from_numpy(np.stack(xs, axis=0)).float(),
        # 将y轴列表转换为PyTorch的张量
        y=torch.from_numpy(np.stack(ys, axis=0)).float(),
        # 将z轴列表转换为PyTorch的张量
        z=torch.from_numpy(np.stack(zs, axis=0)).float(),
        # 设置摄像机的宽度
        width=size,
        # 设置摄像机的高度
        height=size,
        # 设置x方向的视场角
        x_fov=0.7,
        # 设置y方向的视场角
        y_fov=0.7,
        # 设置形状参数，表示1个摄像机和其数量
        shape=(1, len(xs)),
    )
```