# `basic-computer-games\86_Target\csharp\FiringRange.cs`

```
// 使用 Games.Common.Randomness 命名空间中的 Randomness 接口
using Games.Common.Randomness;

// 定义 Target 命名空间
namespace Target
{
    // 定义 FiringRange 类
    internal class FiringRange
    {
        // 声明私有只读字段 _random，类型为 IRandom 接口
        private readonly IRandom _random;
        // 声明私有字段 _targetPosition，类型为 Point
        private Point _targetPosition;

        // 定义 FiringRange 类的构造函数，接受一个 IRandom 类型的参数 random
        public FiringRange(IRandom random)
        {
            // 将参数 random 赋值给字段 _random
            _random = random;
        }

        // 定义 NextTarget 方法，返回值为 Point 类型，用于生成下一个目标位置
        public Point NextTarget() =>  _targetPosition = _random.NextPosition();

        // 定义 Fire 方法，接受 angleFromX、angleFromZ 和 distance 三个参数，返回值为 Explosion 类型
        public Explosion Fire(Angle angleFromX, Angle angleFromZ, float distance)
        {
            // 创建一个新的爆炸位置，使用传入的 angleFromX、angleFromZ 和 distance 参数
            var explosionPosition = new Point(angleFromX, angleFromZ, distance);
            // 计算目标位置与爆炸位置的偏移量
            var targetOffset = explosionPosition - _targetPosition;
            // 返回一个新的 Explosion 对象，使用爆炸位置和目标偏移量作为参数
            return new (explosionPosition, targetOffset);
        }
    }
}
```