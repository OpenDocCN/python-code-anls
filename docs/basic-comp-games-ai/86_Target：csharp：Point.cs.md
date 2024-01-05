# `86_Target\csharp\Point.cs`

```
        # 定义 Point 类，包含私有属性 _angleFromX、_angleFromZ、_x、_y、_z 和 _estimateCount
        # 构造函数，接受 angleFromX、angleFromZ 和 distance 作为参数
        # 将 angleFromX、angleFromZ 和 distance 分别赋值给 _angleFromX、_angleFromZ 和 Distance
```
```python
            _x = distance * Math.Sin(angleFromX.Radians) * Math.Cos(angleFromZ.Radians);
            _y = distance * Math.Sin(angleFromX.Radians) * Math.Sin(angleFromZ.Radians);
            _z = distance * Math.Cos(angleFromX.Radians);
        }
```
```python
        # 定义构造函数内的计算过程，根据给定的 angleFromX、angleFromZ 和 distance 计算出 x、y、z 坐标
```
```python
        public float Distance { get; }
        # 定义 Distance 属性，只读
```
```python
        public int EstimateCount
        {
            get { return _estimateCount; }
            set { _estimateCount = value; }
        }
        # 定义 EstimateCount 属性，可读可写
```
```python
        public float X => _x;
        public float Y => _y;
        public float Z => _z;
        # 定义 X、Y、Z 属性，只读
```
```python
        public float AngleFromX => _angleFromX.Degrees;
        public float AngleFromZ => _angleFromZ.Degrees;
        # 定义 AngleFromX、AngleFromZ 属性，只读，返回角度值
```
```python
        public override string ToString()
        {
            return $"X: {_x}, Y: {_y}, Z: {_z}";
        }
        # 重写 ToString 方法，返回 Point 对象的 x、y、z 坐标信息
    }
}
```
```python
# 定义 Point 类的 ToString 方法，返回 Point 对象的 x、y、z 坐标信息
            _x = distance * (float)Math.Sin(_angleFromZ) * (float)Math.Cos(_angleFromX);  // 计算 x 坐标
            _y = distance * (float)Math.Sin(_angleFromZ) * (float)Math.Sin(_angleFromX);  // 计算 y 坐标
            _z = distance * (float)Math.Cos(_angleFromZ);  // 计算 z 坐标
        }

        public float Distance { get; }  // 获取距离属性

        public float EstimateDistance() =>  // 估算距离
            ++_estimateCount switch  // 递增估算次数并进行判断
            {
                1 => EstimateDistance(20),  // 估算距离，精度为 20
                2 => EstimateDistance(10),  // 估算距离，精度为 10
                3 => EstimateDistance(5),   // 估算距离，精度为 5
                4 => EstimateDistance(1),   // 估算距离，精度为 1
                _ => Distance  // 默认返回实际距离
            };

        public float EstimateDistance(int precision) => (float)Math.Floor(Distance / precision) * precision;  // 根据给定精度估算距离
        // 返回当前点的与 X 轴和 Z 轴的夹角（弧度）
        public string GetBearing() => $"Radians from X axis = {_angleFromX}   from Z axis = {_angleFromZ}";

        // 返回当前点的坐标信息
        public override string ToString() => $"X= {_x}   Y = {_y}   Z= {_z}";

        // 重载减法运算符，实现两个点的坐标相减
        public static Offset operator -(Point p1, Point p2) => new (p1._x - p2._x, p1._y - p2._y, p1._z - p2._z);
    }
}
```