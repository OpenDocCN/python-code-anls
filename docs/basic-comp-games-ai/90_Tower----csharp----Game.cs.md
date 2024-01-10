# `basic-computer-games\90_Tower\csharp\Game.cs`

```
using System;
using Tower.Models;
using Tower.Resources;
using Tower.UI;

namespace Tower
{
    internal class Game
    {
        private readonly Towers _towers;  // 创建私有变量 _towers，用于存储 Towers 类的实例
        private readonly TowerDisplay _display;  // 创建私有变量 _display，用于存储 TowerDisplay 类的实例
        private readonly int _optimalMoveCount;  // 创建私有变量 _optimalMoveCount，用于存储最佳移动次数
        private int _moveCount;  // 创建私有变量 _moveCount，用于存储移动次数

        public Game(int diskCount)  // 创建构造函数，参数为盘子数量
        {
            _towers = new Towers(diskCount);  // 初始化 _towers，创建一个具有指定盘子数量的 Towers 实例
            _display = new TowerDisplay(_towers);  // 初始化 _display，创建一个显示 Towers 实例的 TowerDisplay 实例
            _optimalMoveCount = (1 << diskCount) - 1;  // 计算并存储最佳移动次数
        }

        public bool Play()  // 创建 Play 方法
        {
            Console.Write(Strings.Instructions);  // 在控制台输出游戏指令

            Console.Write(_display);  // 在控制台输出 _display 的内容

            while (true)  // 进入循环
            {
                if (!Input.TryReadNumber(Prompt.Disk, out int disk)) { return false; }  // 尝试读取输入的盘子数量，如果失败则返回 false

                if (!_towers.TryFindDisk(disk, out var from, out var message))  // 尝试在 Towers 实例中找到指定盘子，如果失败
                {
                    Console.WriteLine(message);  // 在控制台输出错误信息
                    continue;  // 继续下一次循环
                }

                if (!Input.TryReadNumber(Prompt.Needle, out var to)) { return false; }  // 尝试读取输入的目标柱编号，如果失败则返回 false

                if (!_towers.TryMoveDisk(from, to))  // 尝试移动盘子，如果失败
                {
                    Console.Write(Strings.IllegalMove);  // 在控制台输出非法移动的提示
                    continue;  // 继续下一次循环
                }

                Console.Write(_display);  // 在控制台输出 _display 的内容

                var result = CheckProgress();  // 调用 CheckProgress 方法并存储返回值
                if (result.HasValue) { return result.Value; }  // 如果返回值存在，则返回该值
            }
        }

        private bool? CheckProgress()  // 创建 CheckProgress 方法
        {
            _moveCount++;  // 移动次数加一

            if (_moveCount == 128)  // 如果移动次数等于 128
            {
                Console.Write(Strings.TooManyMoves);  // 在控制台输出移动次数过多的提示
                return false;  // 返回 false
            }

            if (_towers.Finished)  // 如果游戏已完成
            {
                if (_moveCount == _optimalMoveCount)  // 如果移动次数等于最佳移动次数
                {
                    Console.Write(Strings.Congratulations);  // 在控制台输出祝贺消息
                }

                Console.WriteLine(Strings.TaskFinished, _moveCount);  // 在控制台输出任务完成的消息

                return true;  // 返回 true
            }

            return default;  // 返回默认值
        }
    }
}
```