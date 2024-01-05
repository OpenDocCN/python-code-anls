# `d:/src/tocomm/basic-computer-games\02_Amazing\csharp\Amazing.cs`

```
            # 使用逗号分隔文本，将其分割成字符串数组
            # tokens数组存储了被逗号分隔的文本
            String[] tokens = text.Split(",");

            # 声明一个整型变量val，用于存储转换后的数值
            int val;

            # 尝试将tokens数组中指定位置的字符串转换为整型数值，并将结果存储在val中
            if (Int32.TryParse(tokens[pos], out val))
            {
        // 返回一个随机数，范围在[min, max)之间
        public static int Random(int min, int max)
        {
            // 创建一个随机数生成器对象
            Random random = new Random();
            // 返回[min, max)范围内的随机数
            return random.Next(max - min) + min;
        }

        // 打印"AMAZING PROGRAM"和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"到控制台
        public void Play()
        {
            // 在控制台打印"AMAZING PROGRAM"，并在前面添加28个空格
            Console.WriteLine(Tab(28) + "AMAZING PROGRAM");
            // 在控制台打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加15个空格
            Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            // 输出空行
            Console.WriteLine();

            // 初始化宽度和长度变量
            int width = 0;
            int length = 0;

            // 使用 do-while 循环获取用户输入的宽度和长度，直到输入合法值
            do
            {
                // 获取用户输入的宽度和长度
                String range = DisplayTextAndGetInput("WHAT ARE YOUR WIDTH AND LENGTH");
                // 如果输入包含逗号
                if (range.IndexOf(",") > 0)
                {
                    // 从输入中获取宽度和长度的值
                    width = GetDelimitedValue(range, 0);
                    length = GetDelimitedValue(range, 1);
                }
            }
            while (width < 1 || length < 1);  // 当宽度或长度小于1时继续循环

            // 创建一个网格对象，传入长度和宽度
            Grid grid = new Grid(length, width);
            // 设置入口位置，并将其列索引保存到 enterCol 变量中
            int enterCol = grid.SetupEntrance();

            // 计算总墙数，用于后续操作
            int totalWalls = width * length + 1;
            int count = 2;  // 初始化计数器为2
            Cell cell = grid.StartingCell();  // 获取迷宫的起始单元格

            while (count != totalWalls)  // 当计数器不等于总墙数时循环
            {
                List<Direction> possibleDirs = GetPossibleDirs(grid, cell);  // 获取当前单元格可行的方向列表

                if (possibleDirs.Count != 0)  // 如果可行方向列表不为空
                {
                    cell = SetCellExit(grid, cell, possibleDirs);  // 设置当前单元格的出口，并更新当前单元格
                    cell.Count = count++;  // 更新当前单元格的计数器并递增
                }
                else  // 如果可行方向列表为空
                {
                    cell = grid.GetFirstUnset(cell);  // 获取第一个未设置的单元格
                }
            }
            grid.SetupExit();  // 设置迷宫的出口

            WriteMaze(width, grid, enterCol);  // 将迷宫的信息写入文件
        }

        # 设置细胞的出口方向
        private Cell SetCellExit(Grid grid, Cell cell, List<Direction> possibleDirs)
        {
            # 从可能的方向中随机选择一个方向
            Direction direction = possibleDirs[Random(0, possibleDirs.Count)];
            # 如果方向是向左
            if (direction == Direction.GO_LEFT)
            {
                # 获取前一列的细胞，并将出口类型设置为右出口
                cell = grid.GetPrevCol(cell);
                cell.ExitType = EXIT_RIGHT;
            }
            # 如果方向是向上
            else if (direction == Direction.GO_UP)
            {
                # 获取前一行的细胞，并将出口类型设置为下出口
                cell = grid.GetPrevRow(cell);
                cell.ExitType = EXIT_DOWN;
            }
            # 如果方向是向右
            else if (direction == Direction.GO_RIGHT)
            {
                # 将细胞的出口类型增加右出口，并获取下一列的细胞
                cell.ExitType = cell.ExitType + EXIT_RIGHT;
                cell = grid.GetNextCol(cell);
            }
            else if (direction == Direction.GO_DOWN)  # 如果方向是向下
            {
                cell.ExitType = cell.ExitType + EXIT_DOWN;  # 设置当前单元格的出口类型为当前出口类型加上向下的标识
                cell = grid.GetNextRow(cell);  # 获取下一行的单元格
            }
            return cell;  # 返回单元格
        }

        private void WriteMaze(int width, Grid grid, int enterCol)  # 定义一个名为WriteMaze的私有方法，接受宽度、网格和入口列作为参数
        {
            // top line  # 输出迷宫的顶部边界
            for (int i = 0; i < width; i++)  # 循环遍历迷宫的宽度
            {
                if (i == enterCol) Console.Write(".  ");  # 如果当前列是入口列，则输出一个点和空格
                else Console.Write(".--");  # 否则输出一个点和两个短横线
            }
            Console.WriteLine(".");  # 输出换行

            for (int i = 0; i < grid.Length; i++)  # 循环遍历网格的长度
            {
                Console.Write("I");  // 在控制台输出字符"I"
                for (int j = 0; j < grid.Width; j++)  // 循环遍历grid的宽度
                {
                    if (grid.Cells[i,j].ExitType == EXIT_UNSET || grid.Cells[i, j].ExitType == EXIT_DOWN)  // 如果当前单元格的出口类型为未设置或向下
                        Console.Write("  I");  // 在控制台输出两个空格和字符"I"
                    else Console.Write("   ");  // 否则在控制台输出三个空格
                }
                Console.WriteLine();  // 在控制台输出换行
                for (int j = 0; j < grid.Width; j++)  // 再次循环遍历grid的宽度
                {
                    if (grid.Cells[i,j].ExitType == EXIT_UNSET || grid.Cells[i, j].ExitType == EXIT_RIGHT)  // 如果当前单元格的出口类型为未设置或向右
                        Console.Write(":--");  // 在控制台输出":--"
                    else Console.Write(":  ");  // 否则在控制台输出":  "
                }
                Console.WriteLine(".");  // 在控制台输出"."
            }
        }

        private List<Direction> GetPossibleDirs(Grid grid, Cell cell)  // 定义一个返回Direction列表的函数GetPossibleDirs，参数为grid和cell
        {
            # 创建一个空的可能方向列表
            var possibleDirs = new List<Direction>();
            # 遍历 Direction 枚举类型的所有值，并添加到可能方向列表中
            foreach (var val in Enum.GetValues(typeof(Direction)))
            {
                possibleDirs.Add((Direction)val);
            }

            # 如果单元格所在列为第一列或者前一列已经设置过，则移除向左的方向
            if (cell.Col == FIRST_COL || grid.IsPrevColSet(cell))
            {
                possibleDirs.Remove(Direction.GO_LEFT);
            }
            # 如果单元格所在行为第一行或者前一行已经设置过，则移除向上的方向
            if (cell.Row == FIRST_ROW || grid.IsPrevRowSet(cell))
            {
                possibleDirs.Remove(Direction.GO_UP);
            }
            # 如果单元格所在列为最后一列或者下一列已经设置过，则移除向右的方向
            if (cell.Col == grid.LastCol || grid.IsNextColSet(cell))
            {
                possibleDirs.Remove(Direction.GO_RIGHT);
            }
            # 如果单元格所在行为最后一行或者下一行已经设置过，则移除向下的方向
            if (cell.Row == grid.LastRow || grid.IsNextRowSet(cell))
            {
                possibleDirs.Remove(Direction.GO_DOWN);  # 从可能的方向列表中移除向下的方向
            }
            return possibleDirs;  # 返回更新后的可能方向列表
        }

        private String DisplayTextAndGetInput(String text)  # 定义一个方法，用于显示文本并获取用户输入
        {
            Console.WriteLine(text);  # 在控制台上显示传入的文本
            return Console.ReadLine();  # 获取用户在控制台上的输入并返回
        }


        private enum Direction  # 定义一个枚举类型 Direction
        {
            GO_LEFT,  # 左方向
            GO_UP,  # 上方向
            GO_RIGHT,  # 右方向
            GO_DOWN,  # 下方向
        }
        # 定义 Cell 类，包含出口类型、计数、列、行等属性
        class Cell:
            def __init__(self, row, col):
                self.ExitType = EXIT_UNSET
                self.Count = 0
                self.Col = col
                self.Row = row

        # 定义 Grid 类，包含 Cells 属性，用于存储 Cell 对象的二维数组
        class Grid:
            def __init__(self, rows, cols):
                self.Cells = [[Cell(r, c) for c in range(cols)] for r in range(rows)]
            // 定义 LastCol 属性，用于存储最后一列的索引
            public int LastCol { get; set; }
            // 定义 LastRow 属性，用于存储最后一行的索引
            public int LastRow { get; set; }

            // 定义 Width 属性，用于存储网格的宽度，并设置为只读
            public int Width { get; private set; }
            // 定义 Length 属性，用于存储网格的长度，并设置为只读
            public int Length { get; private set; }

            // 定义 enterCol 变量，用于存储进入的列索引

            // 定义 Grid 类的构造函数，接受长度和宽度作为参数
            public Grid(int length, int width)
            {
                // 初始化 LastCol 属性为 width - 1
                LastCol = width - 1;
                // 初始化 LastRow 属性为 length - 1
                LastRow = length - 1;
                // 初始化 Width 属性为 width
                Width = width;
                // 初始化 Length 属性为 length
                Length = length;

                // 初始化 Cells 数组，用于存储网格的单元格
                Cells = new Cell[length,width];
                // 遍历网格的每个单元格
                for (int i = 0; i < length; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        // 在迷宫的每个位置创建一个新的单元格对象
                        this.Cells[i,j] = new Cell(i, j);
                    }
                }
            }

            public int SetupEntrance()
            {
                // 随机选择迷宫的入口列
                this.enterCol = Random(0, Width);
                // 将入口单元格的计数设置为1
                Cells[0, enterCol].Count = 1;
                // 返回入口列的索引
                return this.enterCol;
            }

            public void SetupExit()
            {
                // 随机选择迷宫的出口列
                int exit = Random(0, Width - 1);
                // 将最后一行的出口单元格的出口类型增加1
                Cells[LastRow, exit].ExitType += 1;
            }

            public Cell StartingCell()
            {
                return Cells[0, enterCol];  # 返回指定位置的单元格数据
            }

            public bool IsPrevColSet(Cell cell)
            {
                return 0 != Cells[cell.Row, cell.Col - 1].Count;  # 检查指定单元格的左侧单元格是否有数据
            }

            public bool IsPrevRowSet(Cell cell)
            {
                return 0 != Cells[cell.Row - 1, cell.Col].Count;  # 检查指定单元格的上方单元格是否有数据
            }

            public bool IsNextColSet(Cell cell)
            {
                return 0 != Cells[cell.Row, cell.Col + 1].Count;  # 检查指定单元格的右侧单元格是否有数据
            }

            public bool IsNextRowSet(Cell cell)
            {
                return 0 != Cells[cell.Row + 1, cell.Col].Count;  # 检查指定单元格下方一行的单元格是否有数据，如果有则返回 true，否则返回 false
            }

            public Cell GetPrevCol(Cell cell)
            {
                return Cells[cell.Row, cell.Col - 1];  # 返回指定单元格的左侧单元格
            }

            public Cell GetPrevRow(Cell cell)
            {
                return Cells[cell.Row - 1, cell.Col];  # 返回指定单元格的上方单元格
            }

            public Cell GetNextCol(Cell cell)
            {
                return Cells[cell.Row, cell.Col + 1];  # 返回指定单元格的右侧单元格
            }

            public Cell GetNextRow(Cell cell)
            {
                return Cells[cell.Row + 1, cell.Col];  # 返回指定单元格下方单元格的数据
            }

            public Cell GetFirstUnset(Cell cell)  # 定义一个名为GetFirstUnset的公共方法，接受一个名为cell的参数
            {
                int col = cell.Col;  # 初始化一个整数变量col，赋值为cell的列数
                int row = cell.Row;  # 初始化一个整数变量row，赋值为cell的行数
                Cell newCell;  # 声明一个名为newCell的Cell类型变量
                do  # 开始一个do-while循环
                {
                    if (col != this.LastCol)  # 如果col不等于LastCol属性的值
                    {
                        col++;  # 则col加1
                    }
                    else if (row != this.LastRow)  # 否则如果row不等于LastRow属性的值
                    {
                        row++;  # 则row加1
                        col = 0;  # 并且col重置为0
                    }
                    else  # 如果条件不成立
                    {
                        row = 0;  # 将行数重置为0
                        col = 0;  # 将列数重置为0
                    }
                }
                while ((newCell = Cells[row, col]).Count == 0);  # 当新单元格的计数为0时循环
                return newCell;  # 返回新单元格
            }
        }
    }

    class Program  # 定义一个名为Program的类
    {
        static void Main(string[] args)  # 定义一个名为Main的静态方法，参数为字符串数组
        {
            new AmazingGame().Play();  # 创建一个AmazingGame对象并调用其Play方法
        }
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```