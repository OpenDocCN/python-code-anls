# `basic-computer-games\86_Target\csharp\Angle.cs`

```py
namespace Target
{
    internal class Angle
    {
        // 使用与原始代码相同的精度定义常量
        private const float PI = 3.14159f;
        private const float DegreesPerRadian = 57.296f;

        private readonly float _radians;

        // 私有构造函数，接受弧度值作为参数
        private Angle(float radians) => _radians = radians;

        // 将角度转换为弧度
        public static Angle InDegrees(float degrees) => new (degrees / DegreesPerRadian);
        // 将旋转转换为弧度
        public static Angle InRotations(float rotations) => new (2 * PI * rotations);

        // 隐式转换，将角度转换为浮点数
        public static implicit operator float(Angle angle) => angle._radians;
    }
}
```