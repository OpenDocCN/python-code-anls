# BasicComputerGames源码解析 82

# `88_3-D_Tic-Tac-Toe/csharp/QubicData.cs`

这是一个由数字组成的表格，我们需要从中找出其中的规律。从表格中可以看出，每行都有四个数字，这些数字按照顺序从大到小排列。因此，我们可以将每个数字的位置记录在一个数组中，然后使用这些数组来查找规律。

我们可以使用以下 Python 代码来找到这些数字的规律：

```python
def find_pattern(pattern):
   pattern_array = pattern.split("，")[1:]
   return [int(num) for num in pattern_array]

pattern = "3412"
pattern_array = find_pattern(pattern)

print(pattern_array)
```

根据这个程序，我们得到了一个由数字组成的列表，其中包含了上述表格中数字的位置。因此，我们可以得出结论：上述表格中的数字规律是每行四个数字，这些数字从大到小排列，并且每个数字的位置记录在一个数组中。


```
﻿namespace ThreeDTicTacToe
{
    /// <summary>
    /// Data in this class was originally given by the following DATA section in
    /// the BASIC program:
    ///
    /// 2030 DATA 1,49,52,4,13,61,64,16,22,39,23,38,26,42,27,43
    /// 2040 DATA 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
    /// 2050 DATA 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
    /// 2060 DATA 39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56
    /// 2070 DATA 57,58,59,60,61,62,63,64
    /// 2080 DATA 1,17,33,49,5,21,37,53,9,25,41,57,13,29,45,61
    /// 2090 DATA 2,18,34,50,6,22,38,54,10,26,42,58,14,30,46,62
    /// 2100 DATA 3,19,35,51,7,23,39,55,11,27,43,59,15,31,47,63
    /// 2110 DATA 4,20,36,52,8,24,40,56,12,28,44,60,16,32,48,64
    /// 2120 DATA 1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61
    /// 2130 DATA 2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62
    /// 2140 DATA 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63
    /// 2150 DATA 4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64
    /// 2160 DATA 1,6,11,16,17,22,27,32,33,38,43,48,49,54,59,64
    /// 2170 DATA 13,10,7,4,29,26,23,20,45,42,39,36,61,58,55,52
    /// 2180 DATA 1,21,41,61,2,22,42,62,3,23,43,63,4,24,44,64
    /// 2190 DATA 49,37,25,13,50,38,26,14,51,39,27,15,52,40,28,16
    /// 2200 DATA 1,18,35,52,5,22,39,56,9,26,43,60,13,30,47,64
    /// 2210 DATA 49,34,19,4,53,38,23,8,57,42,27,12,61,46,31,16
    /// 2220 DATA 1,22,43,64,16,27,38,49,4,23,42,61,13,26,39,52
    ///
    /// In short, each number is an index into the board. The data in this class
    /// is zero-indexed, as opposed to the original data which was one-indexed.
    /// </summary>
    internal static class QubicData
    {
        /// <summary>
        /// The corners and centers of the Qubic board. They correspond to the
        ///  following coordinates:
        ///
        /// [
        ///     111, 411, 414, 114, 141, 441, 444, 144,
        ///     222, 323, 223, 322, 232, 332, 233, 333
        /// ]
        /// </summary>
        public static readonly int[] CornersAndCenters = new int[16]
        {
           //     (X)      ( )      ( )      (X)
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 (X)      ( )      ( )      (X)

           //     ( )      ( )      ( )      ( )
           //         ( )      (X)      (X)      ( )
           //             ( )      (X)      (X)      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      (X)      (X)      ( )
           //             ( )      (X)      (X)      ( )
           //                 ( )      ( )      ( )      ( )

           //     (X)      ( )      ( )      (X)
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 (X)      ( )      ( )      (X)

            0,48,51,3,12,60,63,15,21,38,22,37,25,41,26,42
        };

        /// <summary>
        /// A list of all "winning" rows in the Qubic board; that is, sets of
        ///  four spaces that, if filled entirely by the player (or machine),
        ///  would result in a win.
        ///
        /// Each group of four rows in the list corresponds to a plane in the
        ///  cube, and each plane is organized so that the first and last rows
        ///  are on the plane's edges, while the second and third rows are in
        ///  the middle of the plane. The only exception is the last group of
        ///  rows, which contains the corners and centers rather than a plane.
        ///
        /// The order of the rows in this list is key to how the Qubic AI
        ///  decides its next move.
        /// </summary>
        public static readonly int[,] RowsByPlane = new int[76, 4]
        {
           //     (1)      (1)      (1)      (1)
           //         (2)      (2)      (2)      (2)
           //             (3)      (3)      (3)      (3)
           //                 (4)      (4)      (4)      (4)

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           { 0, 1, 2, 3,  },
           { 4, 5, 6, 7,  },
           { 8, 9, 10,11, },
           { 12,13,14,15, },

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     (1)      (1)      (1)      (1)
           //         (2)      (2)      (2)      (2)
           //             (3)      (3)      (3)      (3)
           //                 (4)      (4)      (4)      (4)

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           { 16,17,18,19, },
           { 20,21,22,23, },
           { 24,25,26,27, },
           { 28,29,30,31, },

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     (1)      (1)      (1)      (1)
           //         (2)      (2)      (2)      (2)
           //             (3)      (3)      (3)      (3)
           //                 (4)      (4)      (4)      (4)

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           { 32,33,34,35, },
           { 36,37,38,39, },
           { 40,41,42,43, },
           { 44,45,46,47, },

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     (1)      (1)      (1)      (1)
           //         (2)      (2)      (2)      (2)
           //             (3)      (3)      (3)      (3)
           //                 (4)      (4)      (4)      (4)

           { 48,49,50,51, },
           { 52,53,54,55, },
           { 56,57,58,59, },
           { 60,61,62,63, },

           //     (1)      ( )      ( )      ( )
           //         (2)      ( )      ( )      ( )
           //             (3)      ( )      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           //     (1)      ( )      ( )      ( )
           //         (2)      ( )      ( )      ( )
           //             (3)      ( )      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           //     (1)      ( )      ( )      ( )
           //         (2)      ( )      ( )      ( )
           //             (3)      ( )      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           //     (1)      ( )      ( )      ( )
           //         (2)      ( )      ( )      ( )
           //             (3)      ( )      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           { 0, 16,32,48, },
           { 4, 20,36,52, },
           { 8, 24,40,56, },
           { 12,28,44,60, },

           //     ( )      (1)      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      (3)      ( )      ( )
           //                 ( )      (4)      ( )      ( )

           //     ( )      (1)      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      (3)      ( )      ( )
           //                 ( )      (4)      ( )      ( )

           //     ( )      (1)      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      (3)      ( )      ( )
           //                 ( )      (4)      ( )      ( )

           //     ( )      (1)      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      (3)      ( )      ( )
           //                 ( )      (4)      ( )      ( )

           { 1, 17,33,49, },
           { 5, 21,37,53, },
           { 9, 25,41,57, },
           { 13,29,45,61, },

           //     ( )      ( )      (1)      ( )
           //         ( )      ( )      (2)      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      (4)      ( )

           //     ( )      ( )      (1)      ( )
           //         ( )      ( )      (2)      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      (4)      ( )

           //     ( )      ( )      (1)      ( )
           //         ( )      ( )      (2)      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      (4)      ( )

           //     ( )      ( )      (1)      ( )
           //         ( )      ( )      (2)      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      (4)      ( )

           { 2, 18,34,50, },
           { 6, 22,38,54, },
           { 10,26,42,58, },
           { 14,30,46,62, },

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      ( )      (2)
           //             ( )      ( )      ( )      (3)
           //                 ( )      ( )      ( )      (4)

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      ( )      (2)
           //             ( )      ( )      ( )      (3)
           //                 ( )      ( )      ( )      (4)

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      ( )      (2)
           //             ( )      ( )      ( )      (3)
           //                 ( )      ( )      ( )      (4)

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      ( )      (2)
           //             ( )      ( )      ( )      (3)
           //                 ( )      ( )      ( )      (4)

           { 3, 19,35,51, },
           { 7, 23,39,55, },
           { 11,27,43,59, },
           { 15,31,47,63, },

           //     (1)      ( )      ( )      ( )
           //         (1)      ( )      ( )      ( )
           //             (1)      ( )      ( )      ( )
           //                 (1)      ( )      ( )      ( )

           //     (2)      ( )      ( )      ( )
           //         (2)      ( )      ( )      ( )
           //             (2)      ( )      ( )      ( )
           //                 (2)      ( )      ( )      ( )

           //     (3)      ( )      ( )      ( )
           //         (3)      ( )      ( )      ( )
           //             (3)      ( )      ( )      ( )
           //                 (3)      ( )      ( )      ( )

           //     (4)      ( )      ( )      ( )
           //         (4)      ( )      ( )      ( )
           //             (4)      ( )      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           { 0, 4, 8, 12, },
           { 16,20,24,28, },
           { 32,36,40,44, },
           { 48,52,56,60, },

           //     ( )      (1)      ( )      ( )
           //         ( )      (1)      ( )      ( )
           //             ( )      (1)      ( )      ( )
           //                 ( )      (1)      ( )      ( )

           //     ( )      (2)      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      (2)      ( )      ( )
           //                 ( )      (2)      ( )      ( )

           //     ( )      (3)      ( )      ( )
           //         ( )      (3)      ( )      ( )
           //             ( )      (3)      ( )      ( )
           //                 ( )      (3)      ( )      ( )

           //     ( )      (4)      ( )      ( )
           //         ( )      (4)      ( )      ( )
           //             ( )      (4)      ( )      ( )
           //                 ( )      (4)      ( )      ( )

           { 1, 5, 9, 13, },
           { 17,21,25,29, },
           { 33,37,41,45, },
           { 49,53,57,61, },

           //     ( )      ( )      (1)      ( )
           //         ( )      ( )      (1)      ( )
           //             ( )      ( )      (1)      ( )
           //                 ( )      ( )      (1)      ( )

           //     ( )      ( )      (2)      ( )
           //         ( )      ( )      (2)      ( )
           //             ( )      ( )      (2)      ( )
           //                 ( )      ( )      (2)      ( )

           //     ( )      ( )      (3)      ( )
           //         ( )      ( )      (3)      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      (3)      ( )

           //     ( )      ( )      (4)      ( )
           //         ( )      ( )      (4)      ( )
           //             ( )      ( )      (4)      ( )
           //                 ( )      ( )      (4)      ( )

           { 2, 6, 10,14, },
           { 18,22,26,30, },
           { 34,38,42,46, },
           { 50,54,58,62, },

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      ( )      (1)
           //             ( )      ( )      ( )      (1)
           //                 ( )      ( )      ( )      (1)

           //     ( )      ( )      ( )      (2)
           //         ( )      ( )      ( )      (2)
           //             ( )      ( )      ( )      (2)
           //                 ( )      ( )      ( )      (2)

           //     ( )      ( )      ( )      (3)
           //         ( )      ( )      ( )      (3)
           //             ( )      ( )      ( )      (3)
           //                 ( )      ( )      ( )      (3)

           //     ( )      ( )      ( )      (4)
           //         ( )      ( )      ( )      (4)
           //             ( )      ( )      ( )      (4)
           //                 ( )      ( )      ( )      (4)

           { 3, 7, 11,15, },
           { 19,23,27,31, },
           { 35,39,43,47, },
           { 51,55,59,63, },

           //     (1)      ( )      ( )      ( )
           //         ( )      (1)      ( )      ( )
           //             ( )      ( )      (1)      ( )
           //                 ( )      ( )      ( )      (1)

           //     (2)      ( )      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      ( )      (2)      ( )
           //                 ( )      ( )      ( )      (2)

           //     (3)      ( )      ( )      ( )
           //         ( )      (3)      ( )      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      ( )      (3)

           //     (4)      ( )      ( )      ( )
           //         ( )      (4)      ( )      ( )
           //             ( )      ( )      (4)      ( )
           //                 ( )      ( )      ( )      (4)

           { 0, 5, 10,15, },
           { 16,21,26,31, },
           { 32,37,42,47, },
           { 48,53,58,63, },

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      (1)      ( )
           //             ( )      (1)      ( )      ( )
           //                 (1)      ( )      ( )      ( )

           //     ( )      ( )      ( )      (2)
           //         ( )      ( )      (2)      ( )
           //             ( )      (2)      ( )      ( )
           //                 (2)      ( )      ( )      ( )

           //     ( )      ( )      ( )      (3)
           //         ( )      ( )      (3)      ( )
           //             ( )      (3)      ( )      ( )
           //                 (3)      ( )      ( )      ( )

           //     ( )      ( )      ( )      (4)
           //         ( )      ( )      (4)      ( )
           //             ( )      (4)      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           { 12,9, 6, 3,  },
           { 28,25,22,19, },
           { 44,41,38,35, },
           { 60,57,54,51, },

           //     (1)      (2)      (3)      (4)
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         (1)      (2)      (3)      (4)
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             (1)      (2)      (3)      (4)
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 (1)      (2)      (3)      (4)

           { 0, 20,40,60, },
           { 1, 21,41,61, },
           { 2, 22,42,62, },
           { 3, 23,43,63, },

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 (1)      (2)      (3)      (4)

           //     ( )      ( )      ( )      ( )
           //         ( )      ( )      ( )      ( )
           //             (1)      (2)      (3)      (4)
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         (1)      (2)      (3)      (4)
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           //     (1)      (2)      (3)      (4)
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 ( )      ( )      ( )      ( )

           { 48,36,24,12, },
           { 49,37,25,13, },
           { 50,38,26,14, },
           { 51,39,27,15, },

           //     (1)      ( )      ( )      ( )
           //         (2)      ( )      ( )      ( )
           //             (3)      ( )      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           //     ( )      (1)      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      (3)      ( )      ( )
           //                 ( )      (4)      ( )      ( )

           //     ( )      ( )      (1)      ( )
           //         ( )      ( )      (2)      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      (4)      ( )

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      ( )      (2)
           //             ( )      ( )      ( )      (3)
           //                 ( )      ( )      ( )      (4)

           { 0, 17,34,51, },
           { 4, 21,38,55, },
           { 8, 25,42,59, },
           { 12,29,46,63, },

           //     ( )      ( )      ( )      (1)
           //         ( )      ( )      ( )      (2)
           //             ( )      ( )      ( )      (3)
           //                 ( )      ( )      ( )      (4)

           //     ( )      ( )      (1)      ( )
           //         ( )      ( )      (2)      ( )
           //             ( )      ( )      (3)      ( )
           //                 ( )      ( )      (4)      ( )

           //     ( )      (1)      ( )      ( )
           //         ( )      (2)      ( )      ( )
           //             ( )      (3)      ( )      ( )
           //                 ( )      (4)      ( )      ( )

           //     (1)      ( )      ( )      ( )
           //         (2)      ( )      ( )      ( )
           //             (3)      ( )      ( )      ( )
           //                 (4)      ( )      ( )      ( )

           { 48,33,18,3,  },
           { 52,37,22,7,  },
           { 56,41,26,11, },
           { 60,45,30,15, },

           //     (1)      ( )      ( )      (3)
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 (4)      ( )      ( )      (2)

           //     ( )      ( )      ( )      ( )
           //         ( )      (1)      (3)      ( )
           //             ( )      (4)      (2)      ( )
           //                 ( )      ( )      ( )      ( )

           //     ( )      ( )      ( )      ( )
           //         ( )      (2)      (4)      ( )
           //             ( )      (3)      (1)      ( )
           //                 ( )      ( )      ( )      ( )

           //     (2)      ( )      ( )      (4)
           //         ( )      ( )      ( )      ( )
           //             ( )      ( )      ( )      ( )
           //                 (3)      ( )      ( )      (1)

           { 0, 21,42,63, },
           { 15,26,37,48, },
           { 3, 22,41,60, },
           { 12,25,38,51, },
        };
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `88_3-D_Tic-Tac-Toe/javascript/qubit.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是在网页上打印信息，接收一个字符串参数（`str`）。首先将该字符串通过`document.getElementById()`获取的`output`元素添加到文档中，然后打印出来。

`input()`函数的作用是接收一个包含用户输入字样的`input`元素（`input_element`）和用户输入的字符串（`input_str`）。首先创建一个包含输入字样的`INPUT`元素，设置其`type`属性为`text`，`length`属性为`50`，然后将该元素添加到文档中的`output`元素中，并设置元素的`focus`属性。接着监听该元素的`keydown`事件，当事件处理程序（也就是`input()`函数）接收到用户按键时，处理程序会将用户输入的字符串存储在`input_str`变量中，并打印出来。然后清空`input_str`，并将`print()`函数打印出来的字符串添加到文档中的`output`元素中。


```
// QUBIT
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

This appears to be a SQL query that selects rows from a table called `table_name` and returns the values in the columns for each row. The columns in the table are `column_1`, `column_2`, `column_3`, etc., and the values in each column are stored in the corresponding row of the table.

The query uses a SELECT statement to specify the columns that should be included in the output, and the `FROM` clause specifies the table from which the data should be retrieved. The table is specified using a parameter (`table_name`), which is passed as an argument to the query in the `COPY` statement.

The `WHERE` clause is used to filter the data that is returned. In this case, the WHERE clause does not specify any conditions, so all rows of the table are included in the output.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var xa = [];
var la = [];
var ma = [[],
          [,1,2,3,4],    // 1
          [,5,6,7,8],    // 2
          [,9,10,11,12], // 3
          [,13,14,15,16],    // 4
          [,17,18,19,20],    // 5
          [,21,22,23,24],    // 6
          [,25,26,27,28],    // 7
          [,29,30,31,32],    // 8
          [,33,34,35,36],    // 9
          [,37,38,39,40],    // 10
          [,41,42,43,44],    // 11
          [,45,46,47,48],    // 12
          [,49,50,51,52],    // 13
          [,53,54,55,56],    // 14
          [,57,58,59,60],    // 15
          [,61,62,63,64],    // 16
          [,1,17,33,49], // 17
          [,5,21,37,53],    // 18
          [,9,25,41,57],   // 19
          [,13,29,45,61], // 20
          [,2,18,34,50], // 21
          [,6,22,38,54],    // 22
          [,10,26,42,58],  // 23
          [,14,30,46,62],   // 24
          [,3,19,35,51], // 25
          [,7,23,39,55],    // 26
          [,11,27,43,59],  // 27
          [,15,31,47,63], // 28
          [,4,20,36,52], // 29
          [,8,24,40,56], // 30
          [,12,28,44,60],    // 31
          [,16,32,48,64],    // 32
          [,1,5,9,13],   // 33
          [,17,21,25,29],    // 34
          [,33,37,41,45],    // 35
          [,49,53,57,61],    // 36
          [,2,6,10,14],  // 37
          [,18,22,26,30],    // 38
          [,34,38,42,46],    // 39
          [,50,54,58,62],    // 40
          [,3,7,11,15],  // 41
          [,19,23,27,31],    // 42
          [,35,39,43,47],    // 43
          [,51,55,59,63],    // 44
          [,4,8,12,16],  // 45
          [,20,24,28,32],    // 46
          [,36,40,44,48],    // 47
          [,52,56,60,64],    // 48
          [,1,6,11,16],  // 49
          [,17,22,27,32],    // 50
          [,33,38,43,48],    // 51
          [,49,54,59,64],    // 52
          [,13,10,7,4],  // 53
          [,29,26,23,20],    // 54
          [,45,42,39,36],    // 55
          [,61,58,55,52],    // 56
          [,1,21,41,61], // 57
          [,2,22,42,62], // 58
          [,3,23,43,63], // 59
          [,4,24,44,64], // 60
          [,49,37,25,13],    // 61
          [,50,38,26,14],    // 62
          [,51,39,27,15],    // 63
          [,52,40,28,16],    // 64
          [,1,18,35,52], // 65
          [,5,22,39,56], // 66
          [,9,26,43,60], // 67
          [,13,30,47,64],    // 68
          [,49,34,19,4], // 69
          [,53,38,23,8], // 70
          [,57,42,27,12],    // 71
          [,61,46,31,16],    // 72
          [,1,22,43,64], // 73
          [,16,27,38,49],    // 74
          [,4,23,42,61], // 75
          [,13,26,39,52] // 76
          ];
```

这段代码定义了一个名为show_board的函数，其功能是输出一个9x9的棋盘，并在棋盘中央输出一行"X X X X X X X X"，然后分别按照行和列的编号输出字符串，即"1 4 9 2 3 8 13 61 64 27 43 27 43"，其中"X"代表行，"X"代表列。

在函数内部，首先使用for循环输出9个空格。然后使用for和for循环分别输出两行，每行包含16个字符，其中包含"X"、"O"、" "和" "。

接下来，使用for和for循环分别遍历棋盘的每个交叉点，计算出该交叉点的行列数i和j，然后使用if语句判断该交叉点是否在棋盘中的某个位置，如果在该位置，则使用字符串中该位置的括号内容。

最后，输出最后一行的字符串，并使用print函数将该行输出。


```
var ya = [,1,49,52,4,13,61,64,16,22,39,23,38,26,42,27,43];

function show_board()
{
    for (xx = 1; xx <= 9; xx++)
        print("\n");
    for (i = 1; i <= 4; i++) {
        for (j = 1; j <= 4; j++) {
            str = "";
            for (i1 = 1; i1 <= j; i1++)
                str += "   ";
            for (k = 1; k <= 4; k++) {
                q = 16 * i + 4 * j + k - 20;
                if (xa[q] == 0)
                    str += "( )      ";
                if (xa[q] == 5)
                    str += "(M)      ";
                if (xa[q] == 1)
                    str += "(Y)      ";
                if (xa[q] == 1 / 8)
                    str += "( )      ";
            }
            print(str + "\n");
            print("\n");
        }
        print("\n");
        print("\n");
    }
}

```

这两个函数可能会在处理一个矩阵或者一个二维数组中的元素。在这个例子中，我们并不知道具体是什么在二维数组中，所以我们只能根据代码的功能来推测其作用。

`process_board()`函数的作用是处理一个二维数组中的元素。具体来说，它通过遍历二维数组中的每个元素，检查行是否为8的倍数。如果是，则将该行的元素全部置为0。这个函数可能是为了在某些特定的需求中对二维数组进行处理而设计的。

`check_for_lines()`函数的作用是检查二维数组中是否存在行。具体来说，它通过遍历每一行，检查行中的每个元素是否都为1/8，如果是，则将该行标记为存在行。这个函数可能是为了在某些特定的需求中检查二维数组中是否存在行而设计的。


```
function process_board()
{
    for (i = 1; i <= 64; i++) {
        if (xa[i] == 1 / 8)
            xa[i] = 0;
    }
}

function check_for_lines()
{
    for (s = 1; s <= 76; s++) {
        j1 = ma[s][1];
        j2 = ma[s][2];
        j3 = ma[s][3];
        j4 = ma[s][4];
        la[s] = xa[j1] + xa[j2] + xa[j3] + xa[j4];
    }
}

```

这两段代码都函数的作用是打印出指定移步中机器人的最终移动方向和位置。

第一段代码 `show_square()` 是一个函数，接收一个参数 `m`，它代表机器人的初始位置。函数通过一系列数学计算来确定机器人在每一帧的最终位置，并打印出来。每一帧的处理过程如下：

1. `k1 = Math.floor((m - 1) / 16) + 1;` 将 `m` 的值除以 16，向上取整得到一个整数，然后加 1。
2. `j2 = m - 16 * (k1 - 1);` 将 `k1` 的值代入计算，得到机器人在这一帧的位置。
3. `k2 = Math.floor((j2 - 1) / 4) + 1;` 将 `j2` 的值除以 4，向上取整得到一个整数，然后加 1。
4. `k3 = m - (k1 - 1) * 16 - (k2 - 1) * 4;` 将前面计算得到的值代入，继续计算得到机器人在这一帧的最终位置。
5. `m = k1 * 100 + k2 * 10 + k3;` 将前面计算得到的值作为 `m` 的值，以便打印输出。
6. `print(" " + m + " ");` 输出字符串 `"MACHINE TAKES"` 和机器人在这一帧的位置。

第二段代码 `select_move()` 是一个函数，用于选择机器人的移动方向。函数接收一个参数 `i`，代表当前帧的编号。函数通过判断 `i` % 4 是否等于 1 来决定移动方向是否为顺时针或逆时针，然后判断 `xa[ma[i][j]]` 是否等于 `sa[i][j]`。如果两个条件都满足，函数返回 `true`，否则返回 `false`。函数的处理过程如下：

1. `if (i % 4 <= 1) {` 如果当前帧是奇数帧，则执行以下操作。
2. `a = 1;` 将 `i` 的值赋为 1，表示向右移动 1步。
3. `} else {` 如果当前帧是偶数帧，则执行以下操作。
4. `a = 2;` 将 `i` 的值赋为 2，表示向右移动 2步。
5. `}` 两侧的语句分别用于偶数帧和奇数帧。
6. `for (j = a; j <= 5 - a; j += 5 - 2 * a) {` 用于移动机器人的位置。
7. `if (xa[ma[i][j]] == sa[i][j])` 用于检查机器人在当前位置是否与上一帧相同。
8. `break;` 如果两个位置相同，则跳出循环，避免重复移动。
9. `}` 两侧的语句用于控制循环次数。
10. `if (j > 5 - a)` 如果移动次数超过了 5 减去 `a`，则表明机器人的位置已经到达终点，函数返回 `false`。
11. `xa[ma[i][j]] = s;` 将 `sa[i][j]` 存储到 `xa` 中，以便下一帧使用。
12. `m = ma[i][j];` 记录机器人当前的位置。
13. `print("MACHINE TAKES");` 输出字符串 `"MACHINE TAKES"` 和机器人当前的位置。
14. `show_square(m);` 调用 `show_square()` 函数，将机器人当前的位置打印出来。


```
function show_square(m)
{
    k1 = Math.floor((m - 1) / 16) + 1;
    j2 = m - 16 * (k1 - 1);
    k2 = Math.floor((j2 - 1) / 4) + 1;
    k3 = m - (k1 - 1) * 16 - (k2 - 1) * 4;
    m = k1 * 100 + k2 * 10 + k3;
    print(" " + m + " ");
}

function select_move() {
    if (i % 4 <= 1) {
        a = 1;
    } else {
        a = 2;
    }
    for (j = a; j <= 5 - a; j += 5 - 2 * a) {
        if (xa[ma[i][j]] == s)
            break;
    }
    if (j > 5 - a)
        return false;
    xa[ma[i][j]] = s;
    m = ma[i][j];
    print("MACHINE TAKES");
    show_square(m);
    return true;
}

```

This is a program that plays the board game "Shogi" for two players. It uses AI to play the game by analyzing the user's moves and deciding what to do in response. The AI uses a deep neural network based on the OpenAI Contin三千番大意中学习得到的技术。

The program starts by defining the initial state of the board and the player that is to play. It then enters a loop that plays the game until one of the players wins or the game is a draw.

In each iteration of the loop, the program displays the current state of the board and prompts the player to choose their next move. The program uses the AI to decide what move to make, based on the pieces it can and cannot move.

If the player chooses to not make an move, the program displays a message and loops back to the previous state of the board. If the player chooses to make a move, the program displays the move as a multiple-line text and updates the display to show the current state of the board.

The program also includes a feature that allows the player to see what the AI推荐的操作， regardless of whether it is a good move or not.

Finally, the program prompts the player to want to try another game, and if the player chooses "YES", the program loops back to the initial state of the board and starts a new game.

Overall, the program is designed to be a fun and interactive way for players to experience the board game "Shogi".


```
// Main control section
async function main()
{
    print(tab(33) + "QUBIC\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    while (1) {
        print("DO YOU WANT INSTRUCTIONS");
        str = await input();
        str = str.substr(0, 1);
        if (str == "Y" || str == "N")
            break;
        print("INCORRECT ANSWER.  PLEASE TYPE 'YES' OR 'NO'");
    }
    if (str == "Y") {
        print("\n");
        print("THE GAME IS TIC-TAC-TOE IN A 4 X 4 X 4 CUBE.\n");
        print("EACH MOVE IS INDICATED BY A 3 DIGIT NUMBER, WITH EACH\n");
        print("DIGIT BETWEEN 1 AND 4 INCLUSIVE.  THE DIGITS INDICATE THE\n");
        print("LEVEL, ROW, AND COLUMN, RESPECTIVELY, OF THE OCCUPIED\n");
        print("PLACE.  \n");
        print("\n");
        print("TO PRINT THE PLAYING BOARD, TYPE 0 (ZERO) AS YOUR MOVE.\n");
        print("THE PROGRAM WILL PRINT THE BOARD WITH YOUR MOVES INDI-\n");
        print("CATED WITH A (Y), THE MACHINE'S MOVES WITH AN (M), AND\n");
        print("UNUSED SQUARES WITH A ( ).  OUTPUT IS ON PAPER.\n");
        print("\n");
        print("TO STOP THE PROGRAM RUN, TYPE 1 AS YOUR MOVE.\n");
        print("\n");
        print("\n");
    }
    while (1) {
        for (i = 1; i <= 64; i++)
            xa[i] = 0;
        z = 1;
        print("DO YOU WANT TO MOVE FIRST");
        while (1) {
            str = await input();
            str = str.substr(0, 1);
            if (str == "Y" || str == "N")
                break;
            print("INCORRECT ANSWER.  PLEASE TYPE 'YES' OR 'NO'");
        }
        while (1) {
            while (1) {
                print(" \n");
                print("YOUR MOVE");
                j1 = parseInt(await input());
                if (j1 == 0) {
                    show_board();
                    continue;
                }
                if (j1 == 1)
                    return;
                k1 = Math.floor(j1 / 100);
                j2 = j1 - k1 * 100;
                k2 = Math.floor(j2 / 10);
                k3 = j2 - k2 * 10;
                m = 16 * k1 + 4 * k2 + k3 - 20;
                if (k1 < 1 || k2 < 1 || k3 < 1 || k1 > 4 || k2 > 4 || k3 >> 4) {
                    print("INCORRECT MOVE, RETYPE IT--");
                } else {
                    process_board();
                    if (xa[m] != 0) {
                        print("THAT SQUARE IS USED, TRY AGAIN.\n");
                    } else {
                        break;
                    }
                }
            }
            xa[m] = 1;
            check_for_lines();
            status = 0;
            for (j = 1; j <= 3; j++) {
                for (i = 1; i <= 76; i++) {
                    if (j == 1) {
                        if (la[i] != 4)
                            continue;
                        print("YOU WIN AS FOLLOWS");
                        for (j = 1; j <= 4; j++) {
                            m = ma[i][j];
                            show_square(m);
                        }
                        status = 1;
                        break;
                    }
                    if (j == 2) {
                        if (la[i] != 15)
                            continue;
                        for (j = 1; j <= 4; j++) {
                            m = ma[i][j];
                            if (xa[m] != 0)
                                continue;
                            xa[m] = 5;
                            print("MACHINE MOVES TO ");
                            show_square(m);
                        }
                        print(", AND WINS AS FOLLOWS");
                        for (j = 1; j <= 4; j++) {
                            m = ma[i][j];
                            show_square(m);
                        }
                        status = 1;
                        break;
                    }
                    if (j == 3) {
                        if (la[i] != 3)
                            continue;
                        print("NICE TRY, MACHINE MOVES TO");
                        for (j = 1; j <= 4; j++) {
                            m = ma[i][j];
                            if (xa[m] != 0)
                                continue;
                            xa[m] = 5;
                            show_square(m);
                            status = 2;
                        }
                        break;
                    }
                }
                if (i <= 76)
                    break;
            }
            if (status == 2)
                continue;
            if (status == 1)
                break;
            // x = x; non-useful in original
            i = 1;
            do {
                la[i] = xa[ma[i][1]] + xa[ma[i][2]] + xa[ma[i][3]] + xa[ma[i][4]];
                l = la[i];
                if (l == 10) {
                    for (j = 1; j <= 4; j++) {
                        if (xa[ma[i][j]] == 0)
                            xa[ma[i][j]] = 1 / 8;
                    }
                }
            } while (++i <= 76) ;
            check_for_lines();
            i = 1;
            do {
                if (la[i] == 0.5) {
                    s = 1 / 8;
                    select_move();
                    break;
                }
                if (la[i] == 5 + 3 / 8) {
                    s = 1 / 8;
                    select_move();
                    break;
                }
            } while (++i <= 76) ;
            if (i <= 76)
                continue;

            process_board();

            i = 1;
            do {
                la[i] = xa[ma[i][1]] + xa[ma[i][2]] + xa[ma[i][3]] + xa[ma[i][4]];
                l = la[i];
                if (l == 2) {
                    for (j = 1; j <= 4; j++) {
                        if (xa[ma[i][j]] == 0)
                            xa[ma[i][j]] = 1 / 8;
                    }
                }
            } while (++i <= 76) ;
            check_for_lines();
            i = 1;
            do {
                if (la[i] == 0.5) {
                    s = 1 / 8;
                    select_move();
                    break;
                }
                if (la[i] == 1 + 3 / 8) {
                    s = 1 / 8;
                    select_move();
                    break;
                }
            } while (++i <= 76) ;
            if (i <= 76)
                continue;

            for (k = 1; k <= 18; k++) {
                p = 0;
                for (i = 4 * k - 3; i <= 4 * k; i++) {
                    for (j = 1; j <= 4; j++)
                        p += xa[ma[i][j]];
                }
                if (p == 4 || p == 9) {
                    s = 1 / 8;
                    for (i = 4 * k - 3; i <= 4 * k; i++) {
                        if (select_move())
                            break;
                    }
                    s = 0;
                }
            }
            if (k <= 18)
                continue
            process_board();
            z = 1;
            do {
                if (xa[ya[z]] == 0)
                    break;
            } while (++z < 17) ;
            if (z >= 17) {
                for (i = 1; i <= 64; i++) {
                    if (xa[i] == 0) {
                        xa[i] = 5;
                        m = i;
                        print("MACHINE LIKES");
                        break;
                    }
                }
                if (i > 64) {
                    print("THE GAME IS A DRAW.\n");
                    break;
                }
            } else {
                m = ya[z];
                xa[m] = 5;
                print("MACHINE MOVES TO");
            }
            show_square(m);
        }
        print(" \n");
        print("DO YOU WANT TO TRY ANOTHER GAME");
        while (1) {
            str = await input();
            str = str.substr(0, 1);
            if (str == "Y" || str == "N")
                break;
            print("INCORRECT ANSWER. PLEASE TYPE 'YES' OR 'NO'");
        }
        if (str == "N")
            break;
    }
}

```

这道题是一个简单的 Python 代码，包含一个名为 `main()` 的函数。函数内部没有包含任何函数体，因此它的作用是“没有具体的作用”。这个函数可以在程序中任何地方被调用，例如在 `print()` 函数中或者在置换到数组中时进行初始化。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `88_3-D_Tic-Tac-Toe/python/qubit.py`

这段代码是一个Python程序，它实现了3D Tic Tac Toe游戏。在游戏过程中，玩家可以通过控制鼠标或者键盘输入来做出各种不同类型的移动。

具体来说，这段代码定义了一个名为Move的枚举类型，包含了游戏的各种状态和机器人的不同移动类型。通过在游戏过程中不断地接收玩家的输入，并且将这些输入转换为游戏状态中的各种机器人类型，从而实现了游戏的核心功能。

在游戏过程中，玩家可以通过点击地图上的各个位置，来选择将鼠标指针或键盘输入转换为相应的游戏状态。例如，当玩家点击地图上的某个位置时，程序会接收到一个鼠标点击事件，对应的游戏状态中会包含指向该位置的向量或者坐标，然后程序会检查这个位置是否属于自己所有的棋子，如果是，则游戏胜利，否则进行下一步操作。

另外，当玩家输入为“/move 1 2”时，程序会进行移动操作，这里“1”和“2”是指要移动的棋子所在的行和列，通过这个方式，玩家就可以通过移动来占领游戏中的重要位置。


```
#!/usr/bin/env python3

# Ported from the BASIC source for 3D Tic Tac Toe
# in BASIC Computer Games, by David H. Ahl
# The code originated from Dartmouth College

from enum import Enum
from typing import Optional, Tuple, Union


class Move(Enum):
    """Game status and types of machine move"""

    HUMAN_WIN = 0
    MACHINE_WIN = 1
    DRAW = 2
    MOVES = 3
    LIKES = 4
    TAKES = 5
    GET_OUT = 6
    YOU_FOX = 7
    NICE_TRY = 8
    CONCEDES = 9


```

Based on the above code, it appears that the `LineSensitiveTature` class is a subclass of `SensitiveTature` and provides methods for playing chess.

The `LineSensitiveTature` class has several methods for controlling the behavior of human players, such as `move_diagonals`, which allows humans to move towards a corner square, `block_human_win`, which prevents human players from winning by checking for gibbal traps, and `move_triple`, which allows humans to move to a square with a higher value than the current piece.

It also has methods for preventing human players from using gibbal traps and using the `concedes` method to end a game in a tie.

It should be noted that the `LineSensitiveTature` class should be subclassed of `SensitiveTature` and should have the following methods:
```
   def int_promotion(self, i) -> Tuple[int, int]:
       return (1, i)

   def pieces_promotion(self, i) -> Tuple[Tuple[int, int]]:
       return (())

   def castle_promotion(self, i) -> Tuple[Tuple[int, int]]:
       return (())

   def move_promotion(self, i, promote) -> Tuple[Tuple[int, int]]:
       return (promote, i)
```
The move promotion method should be added to prevent any class members from being converted to the `int` or `Tuple` types.


```
class Player(Enum):
    EMPTY = 0
    HUMAN = 1
    MACHINE = 2


class TicTacToe3D:
    """The game logic for 3D Tic Tac Toe and the machine opponent"""

    def __init__(self) -> None:
        # 4x4x4 board keeps track of which player occupies each place
        # and used by machine to work out its strategy
        self.board = [0] * 64

        # starting move
        self.corners = [0, 48, 51, 3, 12, 60, 63, 15, 21, 38, 22, 37, 25, 41, 26, 42]

        # lines to check for end game
        self.lines = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35],
            [36, 37, 38, 39],
            [40, 41, 42, 43],
            [44, 45, 46, 47],
            [48, 49, 50, 51],
            [52, 53, 54, 55],
            [56, 57, 58, 59],
            [60, 61, 62, 63],
            [0, 16, 32, 48],
            [4, 20, 36, 52],
            [8, 24, 40, 56],
            [12, 28, 44, 60],
            [1, 17, 33, 49],
            [5, 21, 37, 53],
            [9, 25, 41, 57],
            [13, 29, 45, 61],
            [2, 18, 34, 50],
            [6, 22, 38, 54],
            [10, 26, 42, 58],
            [14, 30, 46, 62],
            [3, 19, 35, 51],
            [7, 23, 39, 55],
            [11, 27, 43, 59],
            [15, 31, 47, 63],
            [0, 4, 8, 12],
            [16, 20, 24, 28],
            [32, 36, 40, 44],
            [48, 52, 56, 60],
            [1, 5, 9, 13],
            [17, 21, 25, 29],
            [33, 37, 41, 45],
            [49, 53, 57, 61],
            [2, 6, 10, 14],
            [18, 22, 26, 30],
            [34, 38, 42, 46],
            [50, 54, 58, 62],
            [3, 7, 11, 15],
            [19, 23, 27, 31],
            [35, 39, 43, 47],
            [51, 55, 59, 63],
            [0, 5, 10, 15],
            [16, 21, 26, 31],
            [32, 37, 42, 47],
            [48, 53, 58, 63],
            [12, 9, 6, 3],
            [28, 25, 22, 19],
            [44, 41, 38, 35],
            [60, 57, 54, 51],
            [0, 20, 40, 60],
            [1, 21, 41, 61],
            [2, 22, 42, 62],
            [3, 23, 43, 63],
            [48, 36, 24, 12],
            [49, 37, 25, 13],
            [50, 38, 26, 14],
            [51, 39, 27, 15],
            [0, 17, 34, 51],
            [4, 21, 38, 55],
            [8, 25, 42, 59],
            [12, 29, 46, 63],
            [48, 33, 18, 3],
            [52, 37, 22, 7],
            [56, 41, 26, 11],
            [60, 45, 30, 15],
            [0, 21, 42, 63],
            [15, 26, 37, 48],
            [3, 22, 41, 60],
            [12, 25, 38, 51],
        ]

    def get(self, x, y, z) -> Player:
        m = self.board[4 * (4 * z + y) + x]
        if m == 40:
            return Player.MACHINE
        elif m == 8:
            return Player.HUMAN
        else:
            return Player.EMPTY

    def move_3d(self, x, y, z, player) -> bool:
        m = 4 * (4 * z + y) + x
        return self.move(m, player)

    def move(self, m, player) -> bool:
        if self.board[m] > 1:
            return False

        if player == Player.MACHINE:
            self.board[m] = 40
        else:
            self.board[m] = 8
        return True

    def get_3d_position(self, m) -> Tuple[int, int, int]:
        x = m % 4
        y = (m // 4) % 4
        z = m // 16
        return x, y, z

    def evaluate_lines(self) -> None:
        self.lineValues = [0] * 76
        for j in range(76):
            value = 0
            for k in range(4):
                value += self.board[self.lines[j][k]]
            self.lineValues[j] = value

    def strategy_mark_line(self, i) -> None:
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                self.board[m] = 1

    def clear_strategy_marks(self) -> None:
        for i in range(64):
            if self.board[i] == 1:
                self.board[i] = 0

    def mark_and_move(self, vlow, vhigh, vmove) -> Optional[Tuple[Move, int]]:
        """
        mark lines that can potentially win the game for the human
        or the machine and choose best place to play
        """
        for i in range(76):
            value = 0
            for j in range(4):
                value += self.board[self.lines[i][j]]
            self.lineValues[i] = value
            if vlow <= value < vhigh:
                if value > vlow:
                    return self.move_triple(i)
                self.strategy_mark_line(i)
        self.evaluate_lines()

        for i in range(76):
            value = self.lineValues[i]
            if value == 4 or value == vmove:
                return self.move_diagonals(i, 1)
        return None

    def machine_move(self) -> Union[None, Tuple[Move, int], Tuple[Move, int, int]]:
        """machine works out what move to play"""
        self.clear_strategy_marks()

        self.evaluate_lines()
        for value, event in [
            (32, self.human_win),
            (120, self.machine_win),
            (24, self.block_human_win),
        ]:
            for i in range(76):
                if self.lineValues[i] == value:
                    return event(i)

        m = self.mark_and_move(80, 88, 43)
        if m is not None:
            return m

        self.clear_strategy_marks()

        m = self.mark_and_move(16, 24, 11)
        if m is not None:
            return m

        for k in range(18):
            value = 0
            for i in range(4 * k, 4 * k + 4):
                for j in range(4):
                    value += self.board[self.lines[i][j]]
            if (32 <= value < 40) or (72 <= value < 80):
                for s in [1, 0]:
                    for i in range(4 * k, 4 * k + 4):
                        m = self.move_diagonals(i, s)
                        if m is not None:
                            return m

        self.clear_strategy_marks()

        for y in self.corners:
            if self.board[y] == 0:
                return (Move.MOVES, y)

        for i in range(64):
            if self.board[i] == 0:
                return (Move.LIKES, i)

        return (Move.DRAW, -1)

    def human_win(self, i) -> Tuple[Move, int, int]:
        return (Move.HUMAN_WIN, -1, i)

    def machine_win(self, i) -> Optional[Tuple[Move, int, int]]:
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                return (Move.MACHINE_WIN, m, i)
        return None

    def block_human_win(self, i) -> Optional[Tuple[Move, int]]:
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                return (Move.NICE_TRY, m)
        return None

    def move_triple(self, i) -> Tuple[Move, int]:
        """make two lines-of-3 or prevent human from doing this"""
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 1:
                if self.lineValues[i] < 40:
                    return (Move.YOU_FOX, m)
                else:
                    return (Move.GET_OUT, m)
        return (Move.CONCEDES, -1)

    # choose move in corners or center boxes of square 4x4
    def move_diagonals(self, i, s) -> Optional[Tuple[Move, int]]:
        if 0 < (i % 4) < 3:
            jrange = [1, 2]
        else:
            jrange = [0, 3]
        for j in jrange:
            m = self.lines[i][j]
            if self.board[m] == s:
                return (Move.TAKES, m)
        return None


```

In this program, the machine will play a game of chess. The machine will have the option to move or draw, or to conquer. If the machine chooses to move, it will have to specify the square it is moving by, and the computer will tell it if the move is valid. If the machine chooses to draw, or if the game is a draw, the computer will display the win.

If the machine chooses to conquer, it will specify the square it is taking, and the computer will tell it if the move is valid.

If the player wants to start a new game, the player will have to enter "yes" or "no".

Also, the program will display the move history, the current state of the game and the player's win status.


```
class Qubit:
    def move_code(self, board, m) -> str:
        x, y, z = board.get_3d_position(m)
        return f"{z + 1:d}{y + 1:d}{x + 1:d}"

    def show_win(self, board, i) -> None:
        for m in board.lines[i]:
            print(self.move_code(board, m))

    def show_board(self, board) -> None:
        c = " YM"
        for z in range(4):
            for y in range(4):
                print("   " * y, end="")
                for x in range(4):
                    p = board.get(x, y, z)
                    print(f"({c[p.value]})      ", end="")
                print("\n")
            print("\n")

    def human_move(self, board) -> bool:
        print()
        c = "1234"
        while True:
            h = input("Your move?\n")
            if h == "1":
                return False
            if h == "0":
                self.show_board(board)
                continue
            if (len(h) == 3) and (h[0] in c) and (h[1] in c) and (h[2] in c):
                x = c.find(h[2])
                y = c.find(h[1])
                z = c.find(h[0])
                if board.move_3d(x, y, z, Player.HUMAN):
                    break

                print("That square is used. Try again.")
            else:
                print("Incorrect move. Retype it--")

        return True

    def play(self) -> None:
        print("Qubic\n")
        print("Create Computing Morristown, New Jersey\n\n\n")
        while True:
            c = input("Do you want instructions?\n")
            if len(c) >= 1 and (c[0] in "ynYN"):
                break
            print("Incorrect answer. Please type 'yes' or 'no.")

        c = c.lower()
        if c[0] == "y":
            print("The game is Tic-Tac-Toe in a 4 x 4 x 4 cube.")
            print("Each move is indicated by a 3 digit number, with each")
            print("digit between 1 and 4 inclusive.  The digits indicate the")
            print("level, row, and column, respectively, of the occupied")
            print("place.\n")

            print("To print the playing board, type 0 (zero) as your move.")
            print("The program will print the board with your moves indicated")
            print("with a (Y), the machine's moves with an (M), and")
            print("unused squares with a ( ).\n")

            print("To stop the program run, type 1 as your move.\n\n")

        play_again = True
        while play_again:
            board = TicTacToe3D()

            while True:
                s = input("Do you want to move first?\n")
                if len(s) >= 1 and (s[0] in "ynYN"):
                    break
                print("Incorrect answer. Please type 'yes' or 'no'.")

            skip_human = s[0] in "nN"

            move_text = [
                "Machine moves to",
                "Machine likes",
                "Machine takes",
                "Let's see you get out of this:  Machine moves to",
                "You fox.  Just in the nick of time, machine moves to",
                "Nice try. Machine moves to",
            ]

            while True:
                if not skip_human and not self.human_move(board):
                    break
                skip_human = False

                m = board.machine_move()
                assert m is not None
                if m[0] == Move.HUMAN_WIN:
                    print("You win as follows,")
                    self.show_win(board, m[2])  # type: ignore
                    break
                elif m[0] == Move.MACHINE_WIN:
                    print(
                        "Machine moves to {}, and wins as follows".format(
                            self.move_code(board, m[1])
                        )
                    )
                    self.show_win(board, m[2])  # type: ignore
                    break
                elif m[0] == Move.DRAW:
                    print("The game is a draw.")
                    break
                elif m[0] == Move.CONCEDES:
                    print("Machine concedes this game.")
                    break
                else:
                    print(move_text[m[0].value - Move.MOVES.value])
                    print(self.move_code(board, m[1]))
                    board.move(m[1], Player.MACHINE)

                self.show_board(board)

            print(" ")
            while True:
                x = input("Do you want to try another game\n")
                if len(x) >= 1 and x[0] in "ynYN":
                    break
                print("Incorrect answer. Please Type 'yes' or 'no'.")

            play_again = x[0] in "yY"


```

这段代码是一个Python程序，主要作用是定义了一个名为“__main__”的模块。在这个模块中，定义了一个条件判断语句，即检查当前程序是否作为主程序运行。如果是主程序，则定义了一个名为“game”的对象，并调用其“play”方法来运行游戏。

具体来说，这段代码的作用是：如果当前程序作为主程序运行，那么定义一个名为“game”的对象，并调用其“play”方法来运行游戏。游戏本身是一个复杂的量子模拟器，无法直接运行程序，因此需要使用Qubit类来模拟量子操作。


```
if __name__ == "__main__":
    game = Qubit()
    game.play()

```