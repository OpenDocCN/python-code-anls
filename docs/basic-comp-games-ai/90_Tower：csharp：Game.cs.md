# `d:/src/tocomm/basic-computer-games\90_Tower\csharp\Game.cs`

```
# 导入必要的模块
using System;
using Tower.Models;
using Tower.Resources;
using Tower.UI;

namespace Tower
{
    internal class Game
    {
        # 初始化游戏对象的属性
        private readonly Towers _towers;  # 创建一个名为_towers的Towers对象
        private readonly TowerDisplay _display;  # 创建一个名为_display的TowerDisplay对象
        private readonly int _optimalMoveCount;  # 创建一个名为_optimalMoveCount的整数属性
        private int _moveCount;  # 创建一个名为_moveCount的整数属性

        # 初始化游戏对象的构造函数
        public Game(int diskCount)
        {
            _towers = new Towers(diskCount);  # 使用传入的diskCount参数创建一个Towers对象并赋值给_towers属性
            _display = new TowerDisplay(_towers);  # 使用_towers属性创建一个TowerDisplay对象并赋值给_display属性
            _optimalMoveCount = (1 << diskCount) - 1;  # 根据传入的diskCount计算出最佳移动次数并赋值给_optimalMoveCount属性
        }
        public bool Play()
        {
            // 打印游戏说明
            Console.Write(Strings.Instructions);

            // 打印当前游戏状态
            Console.Write(_display);

            // 游戏循环
            while (true)
            {
                // 读取用户输入的要移动的盘子号码
                if (!Input.TryReadNumber(Prompt.Disk, out int disk)) { return false; }

                // 查找要移动的盘子是否存在，如果不存在则打印错误信息并继续下一轮循环
                if (!_towers.TryFindDisk(disk, out var from, out var message))
                {
                    Console.WriteLine(message);
                    continue;
                }

                // 读取用户输入的目标柱子号码
                if (!Input.TryReadNumber(Prompt.Needle, out var to)) { return false; }

                // 尝试移动盘子，如果移动失败则继续下一轮循环
                if (!_towers.TryMoveDisk(from, to))
                {
                    Console.Write(Strings.IllegalMove);  # 输出非法移动的提示信息
                    continue;  # 继续下一次循环
                }

                Console.Write(_display);  # 输出当前棋盘状态

                var result = CheckProgress();  # 调用CheckProgress()函数，将返回值赋给result变量
                if (result.HasValue) { return result.Value; }  # 如果result有值，则返回该值
            }
        }

        private bool? CheckProgress()  # 定义一个返回可空布尔值的函数CheckProgress()
        {
            _moveCount++;  # 棋盘移动次数加1

            if (_moveCount == 128)  # 如果移动次数等于128
            {
                Console.Write(Strings.TooManyMoves);  # 输出移动次数过多的提示信息
                return false;  # 返回false
            }
            # 如果所有塔都已经完成移动
            if (_towers.Finished)
            {
                # 如果移动次数等于最佳移动次数
                if (_moveCount == _optimalMoveCount)
                {
                    # 输出祝贺消息
                    Console.Write(Strings.Congratulations);
                }
                # 输出任务完成消息和移动次数
                Console.WriteLine(Strings.TaskFinished, _moveCount);
                # 返回true表示任务完成
                return true;
            }
            # 如果还有未完成的塔，返回默认值
            return default;
        }
    }
}
```