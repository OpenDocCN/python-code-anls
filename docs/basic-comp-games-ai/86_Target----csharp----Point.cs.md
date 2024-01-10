# `basic-computer-games\86_Target\csharp\Point.cs`

```
// 引入 System 命名空间
using System;

// 定义 Target 命名空间
namespace Target
{
    // 定义 Point 类
    internal class Point
    {
        // 声明私有只读字段 _angleFromX 和 _angleFromZ
        private readonly float _angleFromX;
        private readonly float _angleFromZ;

        // 声明私有只读字段 _x、_y 和 _z
        private readonly float _x;
        private readonly float _y;
        private readonly float _z;

        // 声明私有字段 _estimateCount
        private int _estimateCount;

        // 定义 Point 类的构造函数，接受 Angle 对象和距离作为参数
        public Point(Angle angleFromX, Angle angleFromZ, float distance)
        {
            // 初始化 _angleFromX、_angleFromZ 和 Distance 字段
            _angleFromX = angleFromX;
            _angleFromZ = angleFromZ;
            Distance = distance;

            // 根据角度和距离计算出 x、y 和 z 坐标
            _x = distance * (float)Math.Sin(_angleFromZ) * (float)Math.Cos(_angleFromX);
            _y = distance * (float)Math.Sin(_angleFromZ) * (float)Math.Sin(_angleFromX);
            _z = distance * (float)Math.Cos(_angleFromZ);
        }

        // 声明 Distance 属性
        public float Distance { get; }

        // 定义 EstimateDistance 方法，返回估计的距离
        public float EstimateDistance() =>
            // 根据 _estimateCount 的值选择不同的精度进行估计
            ++_estimateCount switch
            {
                1 => EstimateDistance(20),
                2 => EstimateDistance(10),
                3 => EstimateDistance(5),
                4 => EstimateDistance(1),
                _ => Distance
            };

        // 定义 EstimateDistance 方法，接受精度参数，返回估计的距离
        public float EstimateDistance(int precision) => (float)Math.Floor(Distance / precision) * precision;

        // 定义 GetBearing 方法，返回与 X 轴和 Z 轴的弧度
        public string GetBearing() => $"Radians from X axis = {_angleFromX}   from Z axis = {_angleFromZ}";

        // 重写 ToString 方法，返回点的坐标信息
        public override string ToString() => $"X= {_x}   Y = {_y}   Z= {_z}";

        // 定义 Point 类的减法运算符重载，返回两点之间的偏移量
        public static Offset operator -(Point p1, Point p2) => new (p1._x - p2._x, p1._y - p2._y, p1._z - p2._z);
    }
}
```