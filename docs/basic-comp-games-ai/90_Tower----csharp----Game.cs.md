# `basic-computer-games\90_Tower\csharp\Game.cs`

```

// 引入所需的命名空间
using System;
using Tower.Models;
using Tower.Resources;
using Tower.UI;

// 定义游戏类
namespace Tower
{
    internal class Game
    {
        // 声明私有字段
        private readonly Towers _towers; // 声明一个 Towers 类型的字段
        private readonly TowerDisplay _display; // 声明一个 TowerDisplay 类型的字段
        private readonly int _optimalMoveCount; // 声明一个整型的字段，用于存储最佳移动次数
        private int _moveCount; // 声明一个整型的字段，用于存储当前移动次数

        // 构造函数，初始化游戏
        public Game(int diskCount)
        {
            _towers = new Towers(diskCount); // 初始化 Towers 对象
            _display = new TowerDisplay(_towers); // 初始化 TowerDisplay 对象
            _optimalMoveCount = (1 << diskCount) - 1; // 计算最佳移动次数
        }

        // 游戏主逻辑
        public bool Play()
        {
            Console.Write(Strings.Instructions); // 输出游戏指令

            Console.Write(_display); // 输出当前游戏状态

            while (true)
            {
                if (!Input.TryReadNumber(Prompt.Disk, out int disk)) { return false; } // 读取用户输入的盘子编号

                if (!_towers.TryFindDisk(disk, out var from, out var message)) // 查找盘子所在的柱子
                {
                    Console.WriteLine(message); // 输出错误信息
                    continue;
                }

                if (!Input.TryReadNumber(Prompt.Needle, out var to)) { return false; } // 读取用户输入的目标柱子编号

                if (!_towers.TryMoveDisk(from, to)) // 尝试移动盘子
                {
                    Console.Write(Strings.IllegalMove); // 输出非法移动信息
                    continue;
                }

                Console.Write(_display); // 输出移动后的游戏状态

                var result = CheckProgress(); // 检查游戏进度
                if (result.HasValue) { return result.Value; } // 如果游戏结束，返回结果
            }
        }

        // 检查游戏进度
        private bool? CheckProgress()
        {
            _moveCount++; // 移动次数加一

            if (_moveCount == 128) // 如果移动次数达到128次
            {
                Console.Write(Strings.TooManyMoves); // 输出移动次数过多的信息
                return false; // 返回游戏失败
            }

            if (_towers.Finished) // 如果游戏结束
            {
                if (_moveCount == _optimalMoveCount) // 如果移动次数等于最佳移动次数
                {
                    Console.Write(Strings.Congratulations); // 输出祝贺信息
                }

                Console.WriteLine(Strings.TaskFinished, _moveCount); // 输出游戏结束信息

                return true; // 返回游戏成功
            }

            return default; // 返回默认值
        }
    }
}

```