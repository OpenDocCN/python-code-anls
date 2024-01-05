# `86_Target\csharp\Angle.cs`

```
namespace Target
{
    internal class Angle
    {
        // Use same precision for constants as original code
        private const float PI = 3.14159f; // 定义常量 PI，表示圆周率
        private const float DegreesPerRadian = 57.296f; // 定义常量 DegreesPerRadian，表示弧度和角度的转换比例

        private readonly float _radians; // 声明私有成员变量 _radians，表示角度的弧度值

        private Angle(float radians) => _radians = radians; // 定义私有构造函数，接受弧度值参数并赋值给 _radians

        public static Angle InDegrees(float degrees) => new (degrees / DegreesPerRadian); // 定义静态方法，接受角度值参数并转换为弧度值，返回 Angle 对象
        public static Angle InRotations(float rotations) => new (2 * PI * rotations); // 定义静态方法，接受旋转次数参数并转换为弧度值，返回 Angle 对象

        public static implicit operator float(Angle angle) => angle._radians; // 定义隐式转换操作符，将 Angle 对象转换为 float 类型的弧度值
    }
}
```