# `basic-computer-games\56_Life_for_Two\java\LifeForTwo.java`

```
import java.util.*;
import java.util.stream.IntStream;

/**
 * Life for Two
 * <p>
 * The original BASIC program uses a grid with an extras border of cells all around,
 * probably to simplify calculations and manipulations. This java program has the exact
 * grid size and instead uses boundary check conditions in the logic.
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class LifeForTwo {

    final static int GRID_SIZE = 5;

    //Pair of offset which when added to the current cell's coordinates,
    // give the coordinates of the neighbours
    final static int[] neighbourCellOffsets = {
            -1, 0,
            1, 0,
            0, -1,
            0, 1,
            -1, -1,
            1, -1,
            -1, 1,
            1, 1
    };

    //The best term that I could come with to describe these numbers was 'masks'
    //They act like indicators to decide which player won the cell. The value is the score of the cell after all the
    // generation calculations.
    final static List<Integer> maskPlayer1 = List.of(3, 102, 103, 120, 130, 121, 112, 111, 12);
    final static List<Integer> maskPlayer2 = List.of(21, 30, 1020, 1030, 1011, 1021, 1003, 1002, 1012);

    // Initialize the grid with all cells set to 0
    private static void initializeGrid(int[][] grid) {
        for (int[] row : grid) {
            Arrays.fill(row, 0);
        }
    }

    // Calculate the scores for each cell in the grid for one generation
    private static void computeCellScoresForOneGen(int[][] grid) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                // Check if the cell is occupied
                if (grid[i][j] >= 100) {
                    // Calculate the score for the occupied cell
                    calculateScoreForOccupiedCell(grid, i, j);
                }
            }
        }
    }
}
    // 计算玩家得分，返回得分对象
    private static Scores calculatePlayersScore(int[][] grid) {
        int m2 = 0;  // 初始化玩家2的得分
        int m3 = 0;  // 初始化玩家3的得分
        for (int i = 0; i < GRID_SIZE; i++) {  // 遍历行
            for (int j = 0; j < GRID_SIZE; j++) {  // 遍历列
                if (grid[i][j] < 3) {  // 如果格子的值小于3
                    grid[i][j] = 0;  // 将格子的值设为0
                } else {
                    if (maskPlayer1.contains(grid[i][j])) {  // 如果格子的值在玩家1的掩码中
                        m2++;  // 玩家2的得分加1
                    } else if (maskPlayer2.contains(grid[i][j])) {  // 如果格子的值在玩家2的掩码中
                        m3++;  // 玩家3的得分加1
                    }
                }
            }
        }
        return new Scores(m2, m3);  // 返回得分对象
    }

    // 重置网格以进行下一代
    private static void resetGridForNextGen(int[][] grid) {
        for (int i = 0; i < GRID_SIZE; i++) {  // 遍历行
            for (int j = 0; j < GRID_SIZE; j++) {  // 遍历列
                if (grid[i][j] < 3) {  // 如果格子的值小于3
                    grid[i][j] = 0;  // 将格子的值设为0
                } else {
                    if (maskPlayer1.contains(grid[i][j])) {  // 如果格子的值在玩家1的掩码中
                        grid[i][j] = 100;  // 将格子的值设为100
                    } else if (maskPlayer2.contains(grid[i][j])) {  // 如果格子的值在玩家2的掩码中
                        grid[i][j] = 1000;  // 将格子的值设为1000
                    } else {
                        grid[i][j] = 0;  // 否则将格子的值设为0
                    }
                }
            }
        }
    }

    // 计算占用格子的得分
    private static void calculateScoreForOccupiedCell(int[][] grid, int i, int j) {
        var b = 1;  // 初始化得分倍数
        if (grid[i][j] > 999) {  // 如果格子的值大于999
            b = 10;  // 将得分倍数设为10
        }
        for (int k = 0; k < 15; k += 2) {  // 遍历邻居格子的偏移量
            // 检查边界
            var neighbourX = i + neighbourCellOffsets[k];  // 计算邻居格子的X坐标
            var neighbourY = j + neighbourCellOffsets[k + 1];  // 计算邻居格子的Y坐标
            if (neighbourX >= 0 && neighbourX < GRID_SIZE &&
                    neighbourY >= 0 && neighbourY < GRID_SIZE) {  // 如果邻居格子在网格范围内
                grid[neighbourX][neighbourY] = grid[neighbourX][neighbourY] + b;  // 将邻居格子的值加上得分倍数
            }

        }
    }
    // 打印整个网格
    private static void printGrid(int[][] grid) {
        // 打印空行
        System.out.println();
        // 打印网格边缘
        printRowEdge();
        // 打印空行
        System.out.println();
        // 遍历网格的行
        for (int i = 0; i < grid.length; i++) {
            // 打印行号
            System.out.printf("%d ", i + 1);
            // 遍历网格的列
            for (int j = 0; j < grid[i].length; j++) {
                // 打印网格中的字符
                System.out.printf(" %c ", mapChar(grid[i][j]));
            }
            // 打印行号
            System.out.printf(" %d", i + 1);
            // 换行
            System.out.println();
        }
        // 打印网格边缘
        printRowEdge();
        // 打印空行
        System.out.println();
    }

    // 打印网格行的边缘
    private static void printRowEdge() {
        // 打印起始边缘
        System.out.print("0 ");
        // 打印网格列号
        IntStream.range(1, GRID_SIZE + 1).forEach(i -> System.out.printf(" %s ", i));
        // 打印结束边缘
        System.out.print(" 0");
    }

    // 将数字映射为字符
    private static char mapChar(int i) {
        // 如果数字为3或100，返回'*'
        if (i == 3 || i == 100) {
            return '*';
        }
        // 如果数字为30或1000，返回'#'
        if (i == 30 || i == 1000) {
            return '#';
        }
        // 其他情况返回空格
        return ' ';
    }

    // 读取合法的坐标
    private static Coordinate readUntilValidCoordinates(Scanner scanner, int[][] grid) {
        // 坐标是否在范围内的标志
        boolean coordinateInRange = false;
        // 初始化坐标
        Coordinate coordinate = null;
        // 循环直到输入合法的坐标
        while (!coordinateInRange) {
            // 读取坐标
            coordinate = readCoordinate(scanner);
            // 检查坐标是否在范围内且对应网格位置是否为空
            if (coordinate.x <= 0 || coordinate.x > GRID_SIZE
                    || coordinate.y <= 0 || coordinate.y > GRID_SIZE
                    || grid[coordinate.x - 1][coordinate.y - 1] != 0) {
                // 如果坐标非法，提示重新输入
                System.out.println("ILLEGAL COORDS. RETYPE");
            } else {
                // 如果坐标合法，设置标志为true
                coordinateInRange = true;
            }
        }
        // 返回合法的坐标
        return coordinate;
    }
    // 从输入的 Scanner 中读取坐标值并返回 Coordinate 对象
    private static Coordinate readCoordinate(Scanner scanner) {
        Coordinate coordinate = null; // 初始化坐标对象
        int x, y; // 定义坐标的 x 和 y 值
        boolean valid = false; // 标记输入是否有效

        System.out.println("X,Y"); // 打印提示信息
        System.out.print("XXXXXX\r"); // 打印特殊字符
        System.out.print("$$$$$$\r"); // 打印特殊字符
        System.out.print("&&&&&&\r"); // 打印特殊字符

        while (!valid) { // 循环直到输入有效
            try {
                System.out.print("? "); // 打印提示信息
                y = scanner.nextInt(); // 读取输入的 y 值
                x = scanner.nextInt(); // 读取输入的 x 值
                valid = true; // 输入有效
                coordinate = new Coordinate(x, y); // 创建坐标对象
            } catch (InputMismatchException e) { // 捕获输入不匹配异常
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE"); // 打印错误信息
                valid = false; // 输入无效
            } finally {
                scanner.nextLine(); // 清空输入行
            }
        }
        return coordinate; // 返回坐标对象
    }

    // 打印游戏介绍信息
    private static void printIntro() {
        System.out.println("                                LIFE2"); // 打印游戏名称
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"); // 打印创意计算公司信息
        System.out.println("\n\n"); // 打印空行

        System.out.println("\tU.B. LIFE GAME"); // 打印游戏名称
    }

    // 定义坐标类
    private static class Coordinate {
        private final int x, y; // 定义坐标的 x 和 y 值

        // 构造方法，初始化坐标对象
        public Coordinate(int x, int y) {
            this.x = x;
            this.y = y;
        }

        // 获取 x 值
        public int getX() {
            return x;
        }

        // 获取 y 值
        public int getY() {
            return y;
        }

        // 重写 toString 方法，返回坐标对象的字符串表示
        @Override
        public String toString() {
            return "Coordinate{" +
                    "x=" + x +
                    ", y=" + y +
                    '}';
        }

        // 重写 equals 方法，判断两个坐标对象是否相等
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Coordinate that = (Coordinate) o;
            return x == that.x && y == that.y;
        }

        // 重写 hashCode 方法，返回坐标对象的哈希值
        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }
    }
    # 定义一个内部类 Scores，用于存储两个玩家的分数
    private static class Scores {
        # 定义两个私有的不可变整型变量，分别表示玩家1和玩家2的分数
        private final int player1Score;
        private final int player2Score;

        # 构造方法，用于初始化玩家1和玩家2的分数
        public Scores(int player1Score, int player2Score) {
            this.player1Score = player1Score;
            this.player2Score = player2Score;
        }

        # 获取玩家1的分数
        public int getPlayer1Score() {
            return player1Score;
        }

        # 获取玩家2的分数
        public int getPlayer2Score() {
            return player2Score;
        }
    }
# 闭合前面的函数定义
```