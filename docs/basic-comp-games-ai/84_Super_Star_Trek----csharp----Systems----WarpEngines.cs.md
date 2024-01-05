# `84_Super_Star_Trek\csharp\Systems\WarpEngines.cs`

```
using System;  // 导入 System 命名空间，包含了常用的基本类和数据类型
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间，包含了与输入输出相关的类
using SuperStarTrek.Commands;  // 导入 SuperStarTrek.Commands 命名空间，包含了与命令相关的类
using SuperStarTrek.Objects;  // 导入 SuperStarTrek.Objects 命名空间，包含了与对象相关的类
using SuperStarTrek.Resources;  // 导入 SuperStarTrek.Resources 命名空间，包含了与资源相关的类
using SuperStarTrek.Space;  // 导入 SuperStarTrek.Space 命名空间，包含了与空间相关的类

namespace SuperStarTrek.Systems  // 声明 SuperStarTrek.Systems 命名空间
{
    internal class WarpEngines : Subsystem  // 声明 WarpEngines 类，继承自 Subsystem 类
    {
        private readonly Enterprise _enterprise;  // 声明私有字段 _enterprise，类型为 Enterprise
        private readonly IReadWrite _io;  // 声明私有字段 _io，类型为 IReadWrite

        internal WarpEngines(Enterprise enterprise, IReadWrite io)  // WarpEngines 类的构造函数，接受 Enterprise 和 IReadWrite 类型的参数
            : base("Warp Engines", Command.NAV, io)  // 调用父类 Subsystem 的构造函数，传入字符串 "Warp Engines"、Command.NAV 和 io 参数
        {
            _enterprise = enterprise;  // 将传入的 enterprise 参数赋值给 _enterprise 字段
            _io = io;  // 将传入的 io 参数赋值给 _io 字段
        }
        // 重写 ExecuteCommandCore 方法，执行指定象限的命令
        protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
        {
            // 尝试从 IO 中读取课程信息，如果成功则继续执行后续操作
            if (_io.TryReadCourse("Course", "   Lt. Sulu", out var course) &&
                // 尝试获取飞行速度因子，如果成功则继续执行后续操作
                TryGetWarpFactor(out var warpFactor) &&
                // 尝试获取移动距离，如果成功则继续执行后续操作
                TryGetDistanceToMove(warpFactor, out var distanceToMove))
            {
                // 克林贡人移动并开火，返回结果
                var result = quadrant.KlingonsMoveAndFire();
                // 如果游戏结束，则返回结果
                if (result.IsGameOver) { return result; }

                // 企业号修复系统
                _enterprise.RepairSystems(warpFactor);
                // 企业号随机改变系统状态
                _enterprise.VaryConditionOfRandomSystem();
                // 企业号移动，并返回经过的时间
                var timeElapsed = _enterprise.Move(course, warpFactor, distanceToMove);

                // 如果企业号停靠在星港，则执行以下操作
                if (_enterprise.IsDocked)
                {
                    // 企业号放下护盾
                    _enterprise.ShieldControl.DropShields();
                    // 企业号加油
                    _enterprise.Refuel();
                    // 企业号光子管道重新装填
                    _enterprise.PhotonTubes.ReplenishTorpedoes();
                }
                _enterprise.Quadrant.Display(Strings.NowEntering);
                # 显示当前进入的象限信息

                return CommandResult.Elapsed(timeElapsed);
                # 返回命令执行结果，包括经过的时间

            }

            return CommandResult.Ok;
            # 如果没有进入上面的条件语句，则返回命令执行结果为“Ok”

        }

        private bool TryGetWarpFactor(out float warpFactor)
        {
            var maximumWarp = IsDamaged ? 0.2f : 8;
            # 如果飞船受损，则最大速度为0.2，否则为8
            if (_io.TryReadNumberInRange("Warp Factor", 0, maximumWarp, out warpFactor))
            {
                return warpFactor > 0;
                # 如果成功读取到合法的warpFactor，则返回true
            }

            _io.WriteLine(
                IsDamaged && warpFactor > maximumWarp
                    ? "Warp engines are damaged.  Maximum speed = warp 0.2"
                    # 如果飞船受损且warpFactor超过最大值，则显示提示信息
            // 尝试获取移动距离，根据给定的warpFactor计算距离，并将结果存储在distanceToTravel中
            private bool TryGetDistanceToMove(float warpFactor, out int distanceToTravel)
            {
                // 根据warpFactor计算移动距离，并四舍五入取整
                distanceToTravel = (int)Math.Round(warpFactor * 8, MidpointRounding.AwayFromZero);
                // 如果计算出的距离小于等于企业号的能量，则返回true
                if (distanceToTravel <= _enterprise.Energy) { return true; }

                // 如果能量不足，则输出相应信息
                _io.WriteLine("Engineering reports, 'Insufficient energy available");
                _io.WriteLine($"                      for maneuvering at warp {warpFactor} !'");

                // 如果距离小于等于企业号的总能量，并且护盾控制未受损，则输出相应信息
                if (distanceToTravel <= _enterprise.TotalEnergy && !_enterprise.ShieldControl.IsDamaged)
                {
                    _io.Write($"Deflector control room acknowledges {_enterprise.ShieldControl.ShieldEnergy} ");
                    _io.WriteLine("units of energy");
                    _io.WriteLine("                         presently deployed to shields.");
                }
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```