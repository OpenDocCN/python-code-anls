# `basic-computer-games\86_Target\csharp\Angle.cs`

```

// 命名空间 Target，包含 Angle 类
namespace Target
{
    // Angle 类，用于表示角度
    internal class Angle
    {
        // 使用与原始代码相同的精度定义常量
        private const float PI = 3.14159f;
        private const float DegreesPerRadian = 57.296f;

        // 只读字段，表示角度的弧度值
        private readonly float _radians;

        // 私有构造函数，接受弧度值作为参数
        private Angle(float radians) => _radians = radians;

        // 静态方法，将角度转换为弧度
        public static Angle InDegrees(float degrees) => new (degrees / DegreesPerRadian);
        // 静态方法，将角度转换为圈数
        public static Angle InRotations(float rotations) => new (2 * PI * rotations);

        // 隐式转换操作符，将 Angle 对象转换为 float 类型的弧度值
        public static implicit operator float(Angle angle) => angle._radians;
    }
}

```