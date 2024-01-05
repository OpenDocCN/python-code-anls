# `86_Target\csharp\Game.cs`

```
using System;  // 导入 System 命名空间
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间

namespace Target  // 声明 Target 命名空间
{
    internal class Game  // 声明 Game 类
    {
        private readonly IReadWrite _io;  // 声明私有只读字段 _io，类型为 IReadWrite 接口
        private readonly FiringRange _firingRange;  // 声明私有只读字段 _firingRange，类型为 FiringRange 类
        private int _shotCount;  // 声明私有整型字段 _shotCount

        public Game(IReadWrite io, FiringRange firingRange)  // 声明 Game 类的构造函数，接受 IReadWrite 类型的参数 io 和 FiringRange 类型的参数 firingRange
        {
            _io = io;  // 将参数 io 赋值给字段 _io
            _firingRange = firingRange;  // 将参数 firingRange 赋值给字段 _firingRange
        }

        public void Play()  // 声明 Play 方法
        {
            _shotCount = 0;  // 将字段 _shotCount 的值设为 0
            # 从射击范围中获取目标
            target = _firingRange.NextTarget()
            # 输出目标的方位角
            _io.WriteLine(target.GetBearing())
            # 输出目标的大致坐标
            _io.WriteLine(f"Target sighted: approximate coordinates:  {target}")

            # 进入循环，持续进行射击
            while (true):
                # 输出估计距离
                _io.WriteLine(f"     Estimated distance: {target.EstimateDistance()}")
                _io.WriteLine()

                # 进行射击，获取爆炸结果
                explosion = Shoot()

                # 如果爆炸结果过于接近，输出信息并结束射击
                if (explosion.IsTooClose):
                    _io.WriteLine("You blew yourself up!!")
                    return

                # 输出爆炸的方位角
                _io.WriteLine(explosion.GetBearing())

                # 如果击中目标，结束射击
                if (explosion.IsHit):
                {
                    # 报告击中目标的距离
                    ReportHit(explosion.DistanceToTarget);
                    # 返回
                    return;
                }

                # 报告未击中目标
                ReportMiss(explosion);
            }
        }

        # 发射炮弹
        private Explosion Shoot()
        {
            # 从输入中读取三个数字，分别是X轴角度偏差、Z轴角度偏差、距离
            var (xDeviation, zDeviation, distance) = _io.Read3Numbers(
                "Input angle deviation from X, angle deviation from Z, distance");
            # 射击次数加一
            _shotCount++;
            # 换行
            _io.WriteLine();

            # 发射炮弹并返回爆炸对象
            return _firingRange.Fire(Angle.InDegrees(xDeviation), Angle.InDegrees(zDeviation), distance);
        }

        # 报告击中目标
        private void ReportHit(float distance)
        {
            // 输出空行
            _io.WriteLine();
            // 输出提示信息，表示目标已经失效
            _io.WriteLine($" * * * HIT * * *   Target is non-functional");
            // 输出空行
            _io.WriteLine();
            // 输出爆炸距离目标的距离
            _io.WriteLine($"Distance of explosion from target was {distance} kilometers.");
            // 输出空行
            _io.WriteLine();
            // 输出完成任务所用的射击次数
            _io.WriteLine($"Mission accomplished in {_shotCount} shots.");
        }

        // 报告未命中的情况
        private void ReportMiss(Explosion explosion)
        {
            // 调用另一个 ReportMiss 方法，传入爆炸来源的位置
            ReportMiss(explosion.FromTarget);
            // 输出爆炸位置的大致位置
            _io.WriteLine($"Approx position of explosion:  {explosion.Position}");
            // 输出爆炸距离目标的距离
            _io.WriteLine($"     Distance from target = {explosion.DistanceToTarget}");
            // 输出空行
            _io.WriteLine();
            // 输出空行
            _io.WriteLine();
            // 输出空行
            _io.WriteLine();
        }

        // 报告未命中的情况，传入目标偏移量
        private void ReportMiss(Offset targetOffset)
        {
            # 调用 ReportMiss 函数，传入目标偏移量的 X 值和对应的描述文字
            ReportMiss(targetOffset.DeltaX, "in front of", "behind");
            # 调用 ReportMiss 函数，传入目标偏移量的 Y 值和对应的描述文字
            ReportMiss(targetOffset.DeltaY, "to left of", "to right of");
            # 调用 ReportMiss 函数，传入目标偏移量的 Z 值和对应的描述文字
            ReportMiss(targetOffset.DeltaZ, "above", "below");
        }

        # ReportMiss 函数，根据偏移量的正负情况输出对应的描述文字
        private void ReportMiss(float delta, string positiveText, string negativeText) =>
            _io.WriteLine(delta >= 0 ? GetOffsetText(positiveText, delta) : GetOffsetText(negativeText, -delta));

        # GetOffsetText 函数，根据描述文字和距离生成描述信息
        private static string GetOffsetText(string text, float distance) => $"Shot {text} target {distance} kilometers.";
    }
}
```