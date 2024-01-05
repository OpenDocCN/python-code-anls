# `d:/src/tocomm/basic-computer-games\56_Life_for_Two\java\LifeForTwo.java`

```
import java.util.*;  // 导入 Java 的工具包
import java.util.stream.IntStream;  // 导入 Java 的流处理工具

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

    final static int GRID_SIZE = 5;  // 定义常量 GRID_SIZE 为 5，表示网格大小

    //Pair of offset which when added to the current cell's coordinates,
    // give the coordinates of the neighbours
    final static int[] neighbourCellOffsets = {  // 定义数组 neighbourCellOffsets
            -1, 0,  // 第一个元素为 -1
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

    public static void main(String[] args) {
        printIntro(); // 调用打印游戏介绍的函数
        Scanner scan = new Scanner(System.in); // 创建一个用于接收用户输入的 Scanner 对象
        scan.useDelimiter("\\D"); // 设置 Scanner 对象的分隔符为非数字字符
        int[][] grid = new int[GRID_SIZE][GRID_SIZE]; // 创建一个二维数组，用于表示游戏的网格

        initializeGrid(grid); // 调用函数初始化游戏网格

        // 读取每个玩家的初始3步移动
        for (int b = 1; b <= 2; b++) {
            System.out.printf("\nPLAYER %d - 3 LIVE PIECES.%n", b); // 打印玩家编号和提示信息
            for (int k1 = 1; k1 <= 3; k1++) {
                var player1Coordinates = readUntilValidCoordinates(scan, grid); // 读取玩家输入的有效坐标
                grid[player1Coordinates.x - 1][player1Coordinates.y - 1] = (b == 1 ? 3 : 30); // 根据玩家编号在网格上标记对应的位置
            }
        }

        printGrid(grid); // 打印更新后的游戏网格

        calculatePlayersScore(grid); // 计算玩家得分，将3和30转换为100和1000

        resetGridForNextGen(grid); // 重置游戏网格，为下一轮游戏做准备
        computeCellScoresForOneGen(grid); // 计算一代中每个单元格的分数
        // 计算玩家得分
        var playerScores = calculatePlayersScore(grid);
        // 重置网格以准备下一轮游戏
        resetGridForNextGen(grid);

        // 初始化游戏结束标志
        boolean gameOver = false;
        // 游戏循环，直到游戏结束
        while (!gameOver) {
            // 打印游戏网格
            printGrid(grid);
            // 如果玩家1和玩家2的得分都为0，则游戏平局
            if (playerScores.getPlayer1Score() == 0 && playerScores.getPlayer2Score() == 0) {
                System.out.println("\nA DRAW");
                gameOver = true;
            } 
            // 如果玩家2的得分为0，则玩家1获胜
            else if (playerScores.getPlayer2Score() == 0) {
                System.out.println("\nPLAYER 1 IS THE WINNER");
                gameOver = true;
            } 
            // 如果玩家1的得分为0，则玩家2获胜
            else if (playerScores.getPlayer1Score() == 0) {
                System.out.println("\nPLAYER 2 IS THE WINNER");
                gameOver = true;
            } 
            // 否则，继续游戏
            else {
                // 提示玩家1输入移动坐标
                System.out.print("PLAYER 1 ");
                Coordinate player1Move = readCoordinate(scan);
                // 提示玩家2输入移动坐标
                System.out.print("PLAYER 2 ");
                Coordinate player2Move = readCoordinate(scan);
                // 如果玩家1和玩家2的移动不相同
                if (!player1Move.equals(player2Move)) {
                    // 在网格中将玩家1的移动位置标记为100
                    grid[player1Move.x - 1][player1Move.y - 1] = 100;
                    // 在网格中将玩家2的移动位置标记为1000
                    grid[player2Move.x - 1][player2Move.y - 1] = 1000;
                }
                // 在原始代码中，当两个玩家选择相同的单元格时，将B赋值为99
                // 用于控制流程
                computeCellScoresForOneGen(grid);
                // 计算玩家得分
                playerScores = calculatePlayersScore(grid);
                // 重置网格以进行下一代
                resetGridForNextGen(grid);
            }
        }

    }

    // 初始化网格
    private static void initializeGrid(int[][] grid) {
        for (int[] row : grid) {
            Arrays.fill(row, 0);
        }
    }
    // 计算每个细胞的得分
    private static void computeCellScoresForOneGen(int[][] grid) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                // 如果细胞的值大于等于100，则计算占据细胞的得分
                if (grid[i][j] >= 100) {
                    calculateScoreForOccupiedCell(grid, i, j);
                }
            }
        }
    }

    // 计算玩家的得分
    private static Scores calculatePlayersScore(int[][] grid) {
        int m2 = 0;
        int m3 = 0;
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                // 如果细胞的值小于3，则将其置为0
                if (grid[i][j] < 3) {
                    grid[i][j] = 0;
                } else {
                    // 如果细胞的值在maskPlayer1中，则m2加1
                    if (maskPlayer1.contains(grid[i][j])) {
                        m2++;
                    } else if (maskPlayer2.contains(grid[i][j])) {  # 如果maskPlayer2包含grid[i][j]的值
                        m3++;  # m3加1
                    }
                }
            }
        }
        return new Scores(m2, m3);  # 返回新的Scores对象，参数为m2和m3
    }

    private static void resetGridForNextGen(int[][] grid) {  # 重置网格以进行下一代
        for (int i = 0; i < GRID_SIZE; i++) {  # 遍历行
            for (int j = 0; j < GRID_SIZE; j++) {  # 遍历列
                if (grid[i][j] < 3) {  # 如果grid[i][j]的值小于3
                    grid[i][j] = 0;  # 将其置为0
                } else {  # 否则
                    if (maskPlayer1.contains(grid[i][j])) {  # 如果maskPlayer1包含grid[i][j]的值
                        grid[i][j] = 100;  # 将其置为100
                    } else if (maskPlayer2.contains(grid[i][j])) {  # 如果maskPlayer2包含grid[i][j]的值
                        grid[i][j] = 1000;  # 将其置为1000
                    } else {  # 否则
    private static void calculateScoreForOccupiedCell(int[][] grid, int i, int j) {
        // 定义变量 b 并初始化为 1
        var b = 1;
        // 如果当前格子的值大于 999，则将 b 的值设为 10
        if (grid[i][j] > 999) {
            b = 10;
        }
        // 循环遍历邻居格子的偏移量
        for (int k = 0; k < 15; k += 2) {
            // 检查邻居格子的边界
            var neighbourX = i + neighbourCellOffsets[k];
            var neighbourY = j + neighbourCellOffsets[k + 1];
            if (neighbourX >= 0 && neighbourX < GRID_SIZE &&
                    neighbourY >= 0 && neighbourY < GRID_SIZE) {
                // 将邻居格子的值增加 b
                grid[neighbourX][neighbourY] = grid[neighbourX][neighbourY] + b;
            }
    private static void printGrid(int[][] grid) {
        // 打印网格
        System.out.println();
        // 打印行边界
        printRowEdge();
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
            System.out.println();
        }
        // 打印行边界
        printRowEdge();
        System.out.println();
    }
    // 打印每行的边界
    private static void printRowEdge() {
        System.out.print("0 "); // 打印左边界
        IntStream.range(1, GRID_SIZE + 1).forEach(i -> System.out.printf(" %s ", i)); // 打印每个格子的编号
        System.out.print(" 0"); // 打印右边界
    }

    // 将数字映射为字符
    private static char mapChar(int i) {
        if (i == 3 || i == 100) {
            return '*'; // 如果数字为3或100，返回'*'
        }
        if (i == 30 || i == 1000) {
            return '#'; // 如果数字为30或1000，返回'#'
        }
        return ' '; // 其他情况返回空格
    }

    // 读取输入，直到输入的坐标在范围内
    private static Coordinate readUntilValidCoordinates(Scanner scanner, int[][] grid) {
        boolean coordinateInRange = false; // 初始化坐标不在范围内
        Coordinate coordinate = null; // 初始化坐标为null
        while (!coordinateInRange) { // 循环直到坐标在范围内
            # 从扫描仪中读取坐标
            coordinate = readCoordinate(scanner);
            # 检查坐标是否合法，如果不合法则提示重新输入
            if (coordinate.x <= 0 || coordinate.x > GRID_SIZE
                    || coordinate.y <= 0 || coordinate.y > GRID_SIZE
                    || grid[coordinate.x - 1][coordinate.y - 1] != 0) {
                System.out.println("ILLEGAL COORDS. RETYPE");
            } else {
                # 如果坐标合法，则将 coordinateInRange 设置为 true
                coordinateInRange = true;
            }
        }
        # 返回合法的坐标
        return coordinate;
    }

    # 从扫描仪中读取坐标
    private static Coordinate readCoordinate(Scanner scanner) {
        Coordinate coordinate = null;
        int x, y;
        boolean valid = false;

        # 提示用户输入 X 和 Y 坐标
        System.out.println("X,Y");
        # 打印 X 坐标的提示
        System.out.print("XXXXXX\r");
        # 打印 Y 坐标的提示
        System.out.print("$$$$$$\r");
        System.out.print("&&&&&&\r");  # 打印字符串"&&&&&&"，并且将光标移动到行首

        while (!valid) {  # 当valid为false时执行循环
            try {
                System.out.print("? ");  # 打印问号
                y = scanner.nextInt();  # 从输入中读取一个整数赋值给y
                x = scanner.nextInt();  # 从输入中读取一个整数赋值给x
                valid = true;  # 将valid设置为true
                coordinate = new Coordinate(x, y);  # 创建一个Coordinate对象
            } catch (InputMismatchException e) {  # 捕获输入不匹配异常
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");  # 打印错误消息
                valid = false;  # 将valid设置为false
            } finally {
                scanner.nextLine();  # 读取输入中的下一行
            }
        }
        return coordinate;  # 返回coordinate对象
    }

    private static void printIntro() {  # 定义一个私有的静态方法printIntro
        # 打印游戏名称和制作公司信息
        System.out.println("                                LIFE2");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");

        # 打印游戏标题
        System.out.println("\tU.B. LIFE GAME");
    }

    # 定义坐标类
    private static class Coordinate {
        # 坐标类的私有属性 x 和 y
        private final int x, y;

        # 坐标类的构造函数
        public Coordinate(int x, int y) {
            this.x = x;
            this.y = y;
        }

        # 获取 x 坐标的方法
        public int getX() {
            return x;
        }

        # 获取 y 坐标的方法
        public int getY() {
            return y;
```
这是一个方法的返回语句，它返回变量y的值。

```
        @Override
        public String toString() {
            return "Coordinate{" +
                    "x=" + x +
                    ", y=" + y +
                    '}';
        }
```
这是一个重写的toString()方法，它返回一个描述坐标对象的字符串，包括x和y的数值。

```
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Coordinate that = (Coordinate) o;
            return x == that.x && y == that.y;
        }
```
这是一个重写的equals()方法，用于比较两个坐标对象是否相等。首先比较引用是否相同，然后比较类型是否相同，最后比较x和y的数值是否相等。

```
        @Override
```
这是一个注解，表示这是一个重写的方法。
        public int hashCode() {
            // 通过Objects类的hash方法计算对象的哈希码，用于在哈希表中存储和查找对象
            return Objects.hash(x, y);
        }
    }

    private static class Scores {
        private final int player1Score;
        private final int player2Score;

        public Scores(int player1Score, int player2Score) {
            // 构造方法，用于初始化Scores对象的player1Score和player2Score属性
            this.player1Score = player1Score;
            this.player2Score = player2Score;
        }

        public int getPlayer1Score() {
            // 返回player1Score属性的值
            return player1Score;
        }

        public int getPlayer2Score() {
            // 返回player2Score属性的值
            return player2Score;
        }
抱歉，给定的代码片段不完整，无法为其添加注释。
```