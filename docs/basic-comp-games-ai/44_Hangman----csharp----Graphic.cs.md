# `44_Hangman\csharp\Graphic.cs`

```
            {
                for (var j = 0; j < Width; j++)
                {
                    _graphic[i, j] = ' ';
                }
            }
        }

        /// <summary>
        /// Draws the hangman graphic based on the number of incorrect guesses.
        /// </summary>
        /// <param name="incorrectGuesses">The number of incorrect guesses made by the player.</param>
        public void Draw(int incorrectGuesses)
        {
            // Draw the base
            _graphic[11, 1] = '_';
            _graphic[11, 2] = '_';
            _graphic[11, 3] = '_';
            _graphic[11, 4] = '_';
            _graphic[11, 5] = '_';
            _graphic[11, 6] = '_';
            _graphic[11, 7] = '_';

            // Draw the pole
            _graphic[0, 3] = '|';
            _graphic[1, 3] = '|';
            _graphic[2, 3] = '|';
            _graphic[3, 3] = '|';
            _graphic[4, 3] = '|';
            _graphic[5, 3] = '|';
            _graphic[6, 3] = '|';
            _graphic[7, 3] = '|';
            _graphic[8, 3] = '|';
            _graphic[9, 3] = '|';
            _graphic[10, 3] = '|';

            // Draw the head
            if (incorrectGuesses > 0)
            {
                _graphic[3, 3] = 'O';
            }

            // Draw the body
            if (incorrectGuesses > 1)
            {
                _graphic[4, 3] = '|';
                _graphic[5, 3] = '|';
                _graphic[6, 3] = '|';
            }

            // Draw the left arm
            if (incorrectGuesses > 2)
            {
                _graphic[4, 2] = '/';
            }

            // Draw the right arm
            if (incorrectGuesses > 3)
            {
                _graphic[4, 4] = '\\';
            }

            // Draw the left leg
            if (incorrectGuesses > 4)
            {
                _graphic[7, 2] = '/';
            }

            // Draw the right leg
            if (incorrectGuesses > 5)
            {
                _graphic[7, 4] = '\\';
            }

            // Print the hangman graphic
            for (var i = 0; i < Height; i++)
            {
                for (var j = 0; j < Width; j++)
                {
                    Console.Write(_graphic[i, j]);
                }
                Console.WriteLine();
            }
        }
    }
}
            {
                // 清空图形数组中的每个元素，用空格代替
                for (var j = 0; j < Width; j++)
                {
                    _graphic[i, j] = ' ';
                }
            }

            // 画垂直线
            for (var i = 0; i < Height; i++)
            {
                // 在图形数组中指定位置画上 'X'
                _graphic[i, 0] = 'X';
            }

            // 画水平线
            for (var i = 0; i < 7; i++)
            {
                // 在图形数组中指定位置画上 'X'
                _graphic[0, i] = 'X';
            }

            // 画绳子
            _graphic[1, 6] = 'X';  // 在图形数组中的特定位置添加字符 'X'
        }

        public void Print()
        {
            for (var i = 0; i < Height; i++)  // 循环遍历图形的高度
            {
                for (var j = 0; j < Width; j++)  // 在每一行中循环遍历图形的宽度
                {
                    Console.Write(_graphic[i, j]);  // 打印图形数组中特定位置的字符
                }

                Console.Write("\n"); // 换行
            }
        }

        public void AddHead()
        {
            _graphic[2, 5] = '-';  // 在图形数组中的特定位置添加字符 '-'
            _graphic[2, 6] = '-';  // 在图形数组中的特定位置添加字符 '-'
            _graphic[2, 7] = '-';  # 在图形数组中的特定位置添加横线
            _graphic[3, 4] = '(';  # 在图形数组中的特定位置添加左括号
            _graphic[3, 5] = '.';  # 在图形数组中的特定位置添加句号
            _graphic[3, 7] = '.';  # 在图形数组中的特定位置添加句号
            _graphic[3, 8] = ')';  # 在图形数组中的特定位置添加右括号
            _graphic[4, 5] = '-';  # 在图形数组中的特定位置添加横线
            _graphic[4, 6] = '-';  # 在图形数组中的特定位置添加横线
            _graphic[4, 7] = '-';  # 在图形数组中的特定位置添加横线
        }

        public void AddBody()
        {
            for (var i = 5; i < 9; i++)  # 循环遍历特定范围内的值
            {
                _graphic[i, 6] = 'X';  # 在图形数组中的特定位置添加字符'X'
            }
        }

        public void AddRightArm()
        {
            for (var i = 3; i < 7; i++)
            {
                _graphic[i, i - 1] = '\\'; // This is the escape character for the back slash.
            }
        }
```
这段代码是一个 for 循环，用于在 _graphic 数组中添加斜杠字符。斜杠字符是用来表示反斜杠的转义字符。

```
        public void AddLeftArm()
        {
            _graphic[3, 10] = '/';
            _graphic[4, 9] = '/';
            _graphic[5, 8] = '/';
            _graphic[6, 7] = '/';
        }
```
这个函数用于在 _graphic 数组中添加左臂的斜杠字符。

```
        public void AddRightLeg()
        {
            _graphic[9, 5] = '/';
            _graphic[10, 4] = '/';
        }
```
这个函数用于在 _graphic 数组中添加右腿的斜杠字符。
# 在图形中添加左腿
public void AddLeftLeg()
{
    _graphic[9, 7] = '\\';  # 在图形数组中的指定位置添加左腿的一部分
    _graphic[10, 8] = '\\';  # 在图形数组中的指定位置添加左腿的另一部分
}

# 在图形中添加右手
public void AddRightHand()
{
    _graphic[2, 2] = '/';  # 在图形数组中的指定位置添加右手
}

# 在图形中添加左手
public void AddLeftHand()
{
    _graphic[2, 10] = '\\';  # 在图形数组中的指定位置添加左手
}

# 在图形中添加右脚
public void AddRightFoot()
{
    _graphic[11, 9] = '\\';  # 在图形数组中的指定位置添加右脚的一部分
    _graphic[11, 10] = '-';  # 在图形数组中的指定位置添加右脚的另一部分
}
        }  # 结束 AddLeftFoot 方法的定义

        public void AddLeftFoot()
        {
            _graphic[11, 3] = '/';  # 在图形数组中的特定位置添加左脚的一部分
            _graphic[11, 2] = '-';  # 在图形数组中的特定位置添加左脚的另一部分
        }
    }  # 结束类的定义
}  # 结束命名空间的定义
```