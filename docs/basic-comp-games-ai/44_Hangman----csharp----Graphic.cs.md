# `basic-computer-games\44_Hangman\csharp\Graphic.cs`

```

using System;

namespace Hangman
{
    /// <summary>
    /// Represents the main "Hangman" graphic.
    /// </summary>
    public class Graphic
    {
        private readonly char[,] _graphic; // 用于存储Hangman图形的二维字符数组
        private const int Width = 12; // 图形的宽度
        private const int Height = 12; // 图形的高度

        public Graphic()
        {
            // 12 x 12 array to represent the graphics.
            _graphic = new char[Height, Width]; // 创建一个12x12的二维字符数组

            // Fill it with empty spaces.
            for (var i = 0; i < Height; i++)
            {
                for (var j = 0; j < Width; j++)
                {
                    _graphic[i, j] = ' '; // 用空格填充数组
                }
            }

            // Draw the vertical line.
            for (var i = 0; i < Height; i++)
            {
                _graphic[i, 0] = 'X'; // 在数组的第一列绘制垂直线
            }

            // Draw the horizontal line.
            for (var i = 0; i < 7; i++)
            {
                _graphic[0, i] = 'X'; // 在数组的第一行绘制水平线
            }

            // Draw the rope.
            _graphic[1, 6] = 'X'; // 在数组指定位置绘制绞索
        }

        public void Print()
        {
            for (var i = 0; i < Height; i++)
            {
                for (var j = 0; j < Width; j++)
                {
                    Console.Write(_graphic[i, j]); // 打印Hangman图形
                }

                Console.Write("\n"); // 换行
            }
        }

        // 以下是为Hangman图形添加不同部分的方法，每个方法都会在数组中绘制相应的部分
        public void AddHead()
        {
            // 绘制头部
        }

        public void AddBody()
        {
            // 绘制身体
        }

        public void AddRightArm()
        {
            // 绘制右手臂
        }

        public void AddLeftArm()
        {
            // 绘制左手臂
        }

        public void AddRightLeg()
        {
            // 绘制右腿
        }

        public void AddLeftLeg()
        {
            // 绘制左腿
        }

        public void AddRightHand()
        {
            // 绘制右手
        }

        public void AddLeftHand()
        {
            // 绘制左手
        }

        public void AddRightFoot()
        {
            // 绘制右脚
        }

        public void AddLeftFoot()
        {
            // 绘制左脚
        }
    }
}

```