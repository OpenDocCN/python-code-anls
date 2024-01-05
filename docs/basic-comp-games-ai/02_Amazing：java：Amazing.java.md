# `02_Amazing\java\Amazing.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Arrays;  // 导入 Arrays 类
import java.util.Random;  // 导入 Random 类
import java.util.Scanner;  // 导入 Scanner 类
import static java.lang.System.in;  // 静态导入 System 类的 in 属性
import static java.lang.System.out;  // 静态导入 System 类的 out 属性

/**
 * Core algorithm copied from amazing.py
 */
public class Amazing {

    final static int FIRST_COL = 0;  // 定义常量 FIRST_COL，表示第一列
    final static int FIRST_ROW = 0;  // 定义常量 FIRST_ROW，表示第一行
    final static int EXIT_UNSET = 0;  // 定义常量 EXIT_UNSET，表示出口未设置
    final static int EXIT_DOWN = 1;  // 定义常量 EXIT_DOWN，表示向下出口
    final static int EXIT_RIGHT = 2;  // 定义常量 EXIT_RIGHT，表示向右出口
    private final Scanner kbScanner;  // 声明私有的 Scanner 对象 kbScanner
    public Amazing() {  // 构造函数 Amazing
        kbScanner = new Scanner(in);  // 初始化 kbScanner 为一个从 System.in 中读取输入的 Scanner 对象
    }  # 结束一个方法或者代码块

    private static int getDelimitedValue(String text, int pos) {  # 定义一个私有的静态方法，接受一个字符串和一个位置参数
        String[] tokens = text.split(",");  # 使用逗号分割字符串，返回一个字符串数组
        try {  # 尝试执行以下代码
            return Integer.parseInt(tokens[pos]);  # 将数组中指定位置的字符串转换为整数并返回
        } catch (Exception ex) {  # 如果出现异常则执行以下代码
            return 0;  # 返回0
        }
    }

    private static String tab(int spaces) {  # 定义一个私有的静态方法，接受一个空格数参数
        char[] spacesTemp = new char[spaces];  # 创建一个指定长度的字符数组
        Arrays.fill(spacesTemp, ' ');  # 用空格填充数组
        return new String(spacesTemp);  # 将字符数组转换为字符串并返回
    }

    public static int random(int min, int max) {  # 定义一个公共的静态方法，接受最小值和最大值参数
        Random random = new Random();  # 创建一个随机数生成器对象
        return random.nextInt(max - min) + min;  # 生成一个介于最小值和最大值之间的随机整数并返回
    }

    # 定义一个公共方法play，用于执行程序的主要功能
    public void play() {
        # 打印出AMAZING PROGRAM
        out.println(tab(28) + "AMAZING PROGRAM");
        # 打印出CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
        out.println(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        # 打印空行
        out.println();

        # 初始化宽度和长度变量
        int width = 0;
        int length = 0;

        # 使用do-while循环获取用户输入的宽度和长度，直到输入有效值为止
        do {
            # 调用displayTextAndGetInput方法显示提示信息并获取用户输入
            String range = displayTextAndGetInput("WHAT ARE YOUR WIDTH AND LENGTH");
            # 如果输入包含逗号，则解析出宽度和长度
            if (range.indexOf(",") > 0) {
                width = getDelimitedValue(range, 0);
                length = getDelimitedValue(range, 1);
            }
        } while (width < 1 || length < 1);

        # 创建一个Grid对象，传入长度和宽度作为参数
        Grid grid = new Grid(length, width);
        # 调用Grid对象的setupEntrance方法设置入口位置，并将结果赋给enterCol变量
        int enterCol = grid.setupEntrance();
        # 计算迷宫中墙的总数
        int totalWalls = width * length + 1;
        # 初始化计数器
        int count = 2;
        # 获取迷宫的起始单元格
        Cell cell = grid.startingCell();

        # 循环直到墙的数量达到总数
        while (count != totalWalls) {
            # 获取当前单元格可行的方向
            ArrayList<Direction> possibleDirs = getPossibleDirs(grid, cell);

            # 如果存在可行方向
            if (possibleDirs.size() != 0) {
                # 设置当前单元格的出口，并更新计数器
                cell = setCellExit(grid, cell, possibleDirs);
                cell.count = count++;
            } else {
                # 如果不存在可行方向，找到第一个未设置出口的单元格
                cell = grid.getFirstUnset(cell);
            }
        }
        # 设置迷宫的出口
        grid.setupExit();

        # 将迷宫写入文件
        writeMaze(width, grid, enterCol);
    }
    // 设置迷宫中的一个单元格的出口方向
    private Cell setCellExit(Grid grid, Cell cell, ArrayList<Direction> possibleDirs) {
        // 从可能的方向中随机选择一个方向
        Direction direction = possibleDirs.get(random(0, possibleDirs.size()));
        // 如果方向是向左
        if (direction == Direction.GO_LEFT) {
            // 获取前一列的单元格，并将出口类型设置为右侧出口
            cell = grid.getPrevCol(cell);
            cell.exitType = EXIT_RIGHT;
        } 
        // 如果方向是向上
        else if (direction == Direction.GO_UP) {
            // 获取前一行的单元格，并将出口类型设置为下方出口
            cell = grid.getPrevRow(cell);
            cell.exitType = EXIT_DOWN;
        } 
        // 如果方向是向右
        else if (direction == Direction.GO_RIGHT) {
            // 将当前单元格的出口类型加上右侧出口，并获取下一列的单元格
            cell.exitType = cell.exitType + EXIT_RIGHT;
            cell = grid.getNextCol(cell);
        } 
        // 如果方向是向下
        else if (direction == Direction.GO_DOWN) {
            // 将当前单元格的出口类型加上下方出口，并获取下一行的单元格
            cell.exitType = cell.exitType + EXIT_DOWN;
            cell = grid.getNextRow(cell);
        }
        // 返回更新后的单元格
        return cell;
    }

    // 将迷宫写入文件
    private void writeMaze(int width, Grid grid, int enterCol) {
        // 顶部边界
        for (int i = 0; i < width; i++) {  // 循环遍历宽度范围内的值
            if (i == enterCol) {  // 如果当前值等于指定的进入列
                out.print(".  ");  // 输出点和空格
            } else {
                out.print(".--");  // 否则输出点和破折号
            }
        }
        out.println('.');  // 输出换行符

        for (Cell[] rows : grid.cells) {  // 遍历二维数组中的每一行
            out.print("I");  // 输出大写字母I
            for (Cell cell : rows) {  // 遍历当前行中的每个单元格
                if (cell.exitType == EXIT_UNSET || cell.exitType == EXIT_DOWN) {  // 如果单元格的出口类型为未设置或向下
                    out.print("  I");  // 输出两个空格和大写字母I
                } else {
                    out.print("   ");  // 否则输出三个空格
                }
            }
            out.println();  // 输出换行符
            for (Cell cell : rows) {  // 再次遍历当前行中的每个单元格
                # 如果细胞的出口类型为未设置或者向右出口，则打印":--"
                if (cell.exitType == EXIT_UNSET || cell.exitType == EXIT_RIGHT) {
                    out.print(":--");
                } else {
                    # 否则打印":  "
                    out.print(":  ");
                }
            }
            # 打印换行符
            out.println(".");
        }
    }

    # 获取细胞可能的方向
    private ArrayList<Direction> getPossibleDirs(Grid grid, Cell cell) {
        # 创建包含所有方向的列表
        ArrayList<Direction> possibleDirs = new ArrayList<>(Arrays.asList(Direction.values()));

        # 如果细胞所在列为第一列，或者前一列已设置，则移除向左的方向
        if (cell.col == FIRST_COL || grid.isPrevColSet(cell)) {
            possibleDirs.remove(Direction.GO_LEFT);
        }
        # 如果细胞所在行为第一行，或者前一行已设置，则移除向上的方向
        if (cell.row == FIRST_ROW || grid.isPrevRowSet(cell)) {
            possibleDirs.remove(Direction.GO_UP);
        }
        # 如果细胞所在列为最后一列，或者下一列已设置，则移除向右的方向
        possibleDirs.remove(Direction.GO_RIGHT);  # 如果当前单元格的列是最后一列或者下一个单元格的行已经设置，则移除向右移动的方向
        }
        if (cell.row == grid.lastRow || grid.isNextRowSet(cell)) {  # 如果当前单元格的行是最后一行或者下一行的单元格已经设置，则移除向下移动的方向
            possibleDirs.remove(Direction.GO_DOWN);
        }
        return possibleDirs;  # 返回剩余可移动的方向列表
    }

    private String displayTextAndGetInput(String text) {  # 显示文本并获取用户输入
        out.print(text);  # 打印文本
        return kbScanner.next();  # 返回用户输入的内容
    }

    enum Direction {  # 定义枚举类型Direction，包括向左、向上、向右、向下四个方向
        GO_LEFT,
        GO_UP,
        GO_RIGHT,
        GO_DOWN,
    }
    public static class Cell {
        int exitType = EXIT_UNSET;  # 设置出口类型为未设置
        int count = 0;  # 初始化计数为0

        int col;  # 列数
        int row;  # 行数

        public Cell(int row, int col) {  # Cell类的构造函数，传入行数和列数
            this.row = row;  # 初始化行数
            this.col = col;  # 初始化列数
        }
    }

    public static class Grid {
        Cell[][] cells;  # 二维数组，存储Cell对象

        int lastCol;  # 最后一列的索引
        int lastRow;  # 最后一行的索引

        int width;  # 网格的宽度
        int enterCol; // 声明一个整型变量 enterCol

        public Grid(int length, int width) { // 定义一个名为 Grid 的构造函数，接受两个整型参数 length 和 width
            this.lastCol = width - 1; // 将 lastCol 属性赋值为 width - 1
            this.lastRow = length - 1; // 将 lastRow 属性赋值为 length - 1
            this.width = width; // 将 width 属性赋值为 width

            this.cells = new Cell[length][width]; // 创建一个名为 cells 的二维数组，大小为 length * width
            for (int i = 0; i < length; i++) { // 循环遍历 length 次
                this.cells[i] = new Cell[width]; // 为 cells[i] 创建一个新的 Cell 数组，大小为 width
                for (int j = 0; j < width; j++) { // 循环遍历 width 次
                    this.cells[i][j] = new Cell(i, j); // 为 cells[i][j] 创建一个新的 Cell 对象，传入 i 和 j 作为参数
                }
            }
        }

        public int setupEntrance() { // 定义一个名为 setupEntrance 的方法，返回一个整型值
            this.enterCol = random(0, this.width); // 将 enterCol 属性赋值为 0 到 this.width 之间的随机数
            cells[0][this.enterCol].count = 1; // 将 cells[0][this.enterCol] 的 count 属性赋值为 1
            return this.enterCol; // 返回 enterCol 属性的值
        }  # 结束一个方法或函数的定义

        public void setupExit() {  # 定义一个名为setupExit的公共方法，无返回值
            int exit = random(0, width - 1);  # 生成一个随机数，赋值给exit变量
            cells[lastRow][exit].exitType += 1;  # 对cells二维数组中指定位置的exitType属性加1
        }

        public Cell startingCell() {  # 定义一个名为startingCell的公共方法，返回类型为Cell
            return cells[0][enterCol];  # 返回cells二维数组中指定位置的元素
        }

        public boolean isPrevColSet(Cell cell) {  # 定义一个名为isPrevColSet的公共方法，返回类型为boolean，参数为cell
            return 0 != cells[cell.row][cell.col - 1].count;  # 返回判断条件的布尔值
        }

        public boolean isPrevRowSet(Cell cell) {  # 定义一个名为isPrevRowSet的公共方法，返回类型为boolean，参数为cell
            return 0 != cells[cell.row - 1][cell.col].count;  # 返回判断条件的布尔值
        }

        public boolean isNextColSet(Cell cell) {  # 定义一个名为isNextColSet的公共方法，返回类型为boolean，参数为cell
        // 检查指定单元格的右侧单元格是否被设置
        public boolean isNextColSet(Cell cell) {
            return 0 != cells[cell.row][cell.col + 1].count;
        }

        // 检查指定单元格的下方单元格是否被设置
        public boolean isNextRowSet(Cell cell) {
            return 0 != cells[cell.row + 1][cell.col].count;
        }

        // 获取指定单元格的左侧单元格
        public Cell getPrevCol(Cell cell) {
            return cells[cell.row][cell.col - 1];
        }

        // 获取指定单元格的上方单元格
        public Cell getPrevRow(Cell cell) {
            return cells[cell.row - 1][cell.col];
        }

        // 获取指定单元格的右侧单元格
        public Cell getNextCol(Cell cell) {
            return cells[cell.row][cell.col + 1];
        }

        // 获取指定单元格的下方单元格
        public Cell getNextRow(Cell cell) {
            return cells[cell.row + 1][cell.col];
        }
        // 返回指定单元格下方单元格的数据
        public Cell getBelow(Cell cell) {
            return cells[cell.row + 1][cell.col];
        }

        // 返回第一个未设置的单元格
        public Cell getFirstUnset(Cell cell) {
            int col = cell.col;
            int row = cell.row;
            Cell newCell;
            // 循环直到找到未设置的单元格
            do {
                if (col != this.lastCol) {
                    col++;
                } else if (row != this.lastRow) {
                    row++;
                    col = 0;
                } else {
                    row = 0;
                    col = 0;
                }
            } while ((newCell = cells[row][col]).count == 0); // 判断单元格是否已设置
            return newCell;
        }
    }
```

这部分代码是一个缩进错误，应该删除这两行代码。
```