# `D:\src\scipysrc\matplotlib\src\py_converters_11.cpp`

```py
// 包含名为 "py_converters_11.h" 的头文件，这里假设包含了必要的声明和定义

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine)
{
    // 如果 transform 是 None，则假定为单位变换，不改变 affine
    if (transform.is_none()) {
        return;
    }

    // 尝试将 transform 转换为双精度的二维 C 风格数组
    auto array = py::array_t<double, py::array::c_style>::ensure(transform);
    // 如果转换失败，或者数组维度不是 2，或者尺寸不是 3x3，则抛出异常
    if (!array || array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3) {
        throw std::invalid_argument("Invalid affine transformation matrix");
    }

    // 获取数组的数据指针
    auto buffer = array.data();
    // 将数组中的数据赋值给 affine 对象的相应成员变量
    affine.sx = buffer[0];   // 缩放因子 x
    affine.shx = buffer[1];  // 剪切因子 x
    affine.tx = buffer[2];   // 平移因子 x
    affine.shy = buffer[3];  // 剪切因子 y
    affine.sy = buffer[4];   // 缩放因子 y
    affine.ty = buffer[5];   // 平移因子 y
}
```