# `basic-computer-games\02_Amazing\java\Amazing.java`

```
# 导入所需的类
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
# 导入静态成员变量
import static java.lang.System.in;
import static java.lang.System.out;

/**
 * 从amazing.py中复制的核心算法
 */
public class Amazing {

    # 定义常量
    final static int FIRST_COL = 0;
    final static int FIRST_ROW = 0;
    final static int EXIT_UNSET = 0;
    final static int EXIT_DOWN = 1;
    final static int EXIT_RIGHT = 2;
    # 声明Scanner对象
    private final Scanner kbScanner;
    # 构造函数
    public Amazing() {
        kbScanner = new Scanner(in);
    }

    # 解析文本并获取指定位置的值
    private static int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        try {
            return Integer.parseInt(tokens[pos]);
        } catch (Exception ex) {
            return 0;
        }
    }

    # 生成指定数量的空格字符串
    private static String tab(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    # 生成指定范围内的随机数
    public static int random(int min, int max) {
        Random random = new Random();
        return random.nextInt(max - min) + min;
    }
}
    // 输出"AMAZING PROGRAM"，并且居中显示
    public void play() {
        out.println(tab(28) + "AMAZING PROGRAM");
        // 输出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并且居中显示
        out.println(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        out.println();

        // 初始化宽度和长度
        int width = 0;
        int length = 0;

        // 循环直到输入的宽度和长度都大于等于1
        do {
            // 获取用户输入的宽度和长度
            String range = displayTextAndGetInput("WHAT ARE YOUR WIDTH AND LENGTH");
            // 如果输入包含逗号
            if (range.indexOf(",") > 0) {
                // 获取逗号分隔的宽度和长度
                width = getDelimitedValue(range, 0);
                length = getDelimitedValue(range, 1);
            }
        } while (width < 1 || length < 1);

        // 创建一个网格对象
        Grid grid = new Grid(length, width);
        // 设置入口的列
        int enterCol = grid.setupEntrance();

        // 计算总墙数
        int totalWalls = width * length + 1;
        int count = 2;
        // 获取起始单元格
        Cell cell = grid.startingCell();

        // 循环直到墙的数量达到总墙数
        while (count != totalWalls) {
            // 获取可能的方向
            ArrayList<Direction> possibleDirs = getPossibleDirs(grid, cell);

            // 如果有可能的方向
            if (possibleDirs.size() != 0) {
                // 设置单元格的出口
                cell = setCellExit(grid, cell, possibleDirs);
                cell.count = count++;
            } else {
                // 获取第一个未设置的单元格
                cell = grid.getFirstUnset(cell);
            }
        }
        // 设置出口
        grid.setupExit();

        // 写入迷宫
        writeMaze(width, grid, enterCol);
    }

    // 设置单元格的出口
    private Cell setCellExit(Grid grid, Cell cell, ArrayList<Direction> possibleDirs) {
        // 随机选择一个可能的方向
        Direction direction = possibleDirs.get(random(0, possibleDirs.size()));
        // 根据方向设置单元格的出口类型，并移动到相应的单元格
        if (direction == Direction.GO_LEFT) {
            cell = grid.getPrevCol(cell);
            cell.exitType = EXIT_RIGHT;
        } else if (direction == Direction.GO_UP) {
            cell = grid.getPrevRow(cell);
            cell.exitType = EXIT_DOWN;
        } else if (direction == Direction.GO_RIGHT) {
            cell.exitType = cell.exitType + EXIT_RIGHT;
            cell = grid.getNextCol(cell);
        } else if (direction == Direction.GO_DOWN) {
            cell.exitType = cell.exitType + EXIT_DOWN;
            cell = grid.getNextRow(cell);
        }
        return cell;
    }
    // 将迷宫以文本形式输出，包括墙壁和通道
    private void writeMaze(int width, Grid grid, int enterCol) {
        // 输出迷宫顶部的墙壁
        for (int i = 0; i < width; i++) {
            if (i == enterCol) {
                out.print(".  ");
            } else {
                out.print(".--");
            }
        }
        out.println('.');

        // 遍历迷宫的每一行
        for (Cell[] rows : grid.cells) {
            out.print("I");
            // 输出迷宫每个单元格的左右墙壁
            for (Cell cell : rows) {
                if (cell.exitType == EXIT_UNSET || cell.exitType == EXIT_DOWN) {
                    out.print("  I");
                } else {
                    out.print("   ");
                }
            }
            out.println();
            // 输出迷宫每个单元格的上下墙壁
            for (Cell cell : rows) {
                if (cell.exitType == EXIT_UNSET || cell.exitType == EXIT_RIGHT) {
                    out.print(":--");
                } else {
                    out.print(":  ");
                }
            }
            out.println(".");
        }
    }

    // 获取指定单元格的可能移动方向
    private ArrayList<Direction> getPossibleDirs(Grid grid, Cell cell) {
        ArrayList<Direction> possibleDirs = new ArrayList<>(Arrays.asList(Direction.values()));

        // 如果单元格在第一列或者左侧有墙壁，则移除向左移动的方向
        if (cell.col == FIRST_COL || grid.isPrevColSet(cell)) {
            possibleDirs.remove(Direction.GO_LEFT);
        }
        // 如果单元格在第一行或者上方有墙壁，则移除向上移动的方向
        if (cell.row == FIRST_ROW || grid.isPrevRowSet(cell)) {
            possibleDirs.remove(Direction.GO_UP);
        }
        // 如果单元格在最后一列或者右侧有墙壁，则移除向右移动的方向
        if (cell.col == grid.lastCol || grid.isNextColSet(cell)) {
            possibleDirs.remove(Direction.GO_RIGHT);
        }
        // 如果单元格在最后一行或者下方有墙壁，则移除向下移动的方向
        if (cell.row == grid.lastRow || grid.isNextRowSet(cell)) {
            possibleDirs.remove(Direction.GO_DOWN);
        }
        return possibleDirs;
    }

    // 显示文本并获取用户输入
    private String displayTextAndGetInput(String text) {
        out.print(text);
        return kbScanner.next();
    }

    // 移动方向的枚举类型
    enum Direction {
        GO_LEFT,
        GO_UP,
        GO_RIGHT,
        GO_DOWN,
    }
    # 定义一个静态内部类 Cell，用于表示迷宫中的一个单元格
    public static class Cell {
        # 定义单元格的出口类型，默认为未设置
        int exitType = EXIT_UNSET;
        # 定义单元格的计数，默认为0
        int count = 0;

        # 定义单元格的列号
        int col;
        # 定义单元格的行号
        int row;

        # 构造方法，初始化单元格的行号和列号
        public Cell(int row, int col) {
            this.row = row;
            this.col = col;
        }
    }
    # 代码块结束
# 闭合前面的函数定义
```