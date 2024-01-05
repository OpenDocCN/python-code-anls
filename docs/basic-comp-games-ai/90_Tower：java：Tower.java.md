# `90_Tower\java\Tower.java`

```
import java.lang.Math;  // 导入 java.lang.Math 类，用于执行数学运算
import java.util.Scanner;  // 导入 java.util.Scanner 类，用于接收用户输入

/**
 * Game of Tower
 * <p>
 * Based on the BASIC game of Tower here
 * https://github.com/coding-horror/basic-computer-games/blob/main/90%20Tower/tower.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Tower {

  private final static int MAX_DISK_SIZE = 15;  // 定义最大盘子数量为 15

  private final static int MAX_NUM_COLUMNS = 3;  // 定义最大柱子数量为 3
  private final static int MAX_NUM_MOVES = 128;  // 定义最大移动次数为128

  private final static int MAX_NUM_ROWS = 7;  // 定义最大行数为7

  private final Scanner scan;  // 用于用户输入的扫描器

  // 代表所有可能的磁盘位置
  private int[][] positions;

  private enum Step {
    INITIALIZE, SELECT_TOTAL_DISKS, SELECT_DISK_MOVE, SELECT_NEEDLE, CHECK_SOLUTION  // 定义了一个步骤枚举类型，包括初始化、选择总磁盘数、选择磁盘移动、选择针、检查解决方案
  }


  public Tower() {

    scan = new Scanner(System.in);  // 初始化扫描器，用于从控制台获取输入

    // Row 0 and column 0 are not used
```

    # 创建一个二维数组用于存储位置信息，行数为 MAX_NUM_ROWS + 1，列数为 MAX_NUM_COLUMNS + 1
    positions = new int[MAX_NUM_ROWS + 1][MAX_NUM_COLUMNS + 1];

  }  // End of constructor Tower


  public class Position {

    public int row;
    public int column;

    # Position 类的构造函数，用于初始化行和列的位置信息
    public Position(int row, int column) {
      this.row = row;
      this.column = column;

    }  // End of constructor Position

  }  // End of inner class Position


  # play 方法用于执行游戏的逻辑
  public void play() {
    showIntro();  # 调用showIntro()方法，显示游戏介绍
    startGame();  # 调用startGame()方法，开始游戏

  }  # 方法play结束


  private void showIntro() {

    System.out.println(" ".repeat(32) + "TOWERS");  # 在控制台打印"TOWERS"，并在前面添加32个空格
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 在控制台打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加14个空格
    System.out.println("\n\n");  # 在控制台打印两个换行符

  }  # 方法showIntro结束


  private void startGame() {

    boolean diskMoved = false;  # 创建一个布尔变量diskMoved，并初始化为false

    int column = 0; // 初始化列数为0
    int disk = 0; // 初始化磁盘数为0
    int needle = 0; // 初始化针数为0
    int numDisks = 0; // 初始化磁盘数量为0
    int numErrors = 0; // 初始化错误数量为0
    int numMoves = 0; // 初始化移动次数为0
    int row = 0; // 初始化行数为0

    Step nextStep = Step.INITIALIZE; // 初始化下一步操作为初始化

    String userResponse = ""; // 初始化用户响应为空字符串

    Position diskPosition = new Position(0, 0); // 初始化磁盘位置为(0, 0)

    // 开始外部循环
    while (true) {

      switch (nextStep) { // 根据下一步操作进行切换
        case INITIALIZE:

          // Initialize error count
          numErrors = 0;  // 初始化错误计数

          // Initialize positions
          for (row = 1; row <= MAX_NUM_ROWS; row++) {
            for (column = 1; column <= MAX_NUM_COLUMNS; column++) {
              positions[row][column] = 0;  // 初始化位置数组，将所有位置的值设为0
            }
          }

          // Display description
          System.out.println("");
          System.out.println("TOWERS OF HANOI PUZZLE.\n");  // 显示游戏描述
          System.out.println("YOU MUST TRANSFER THE DISKS FROM THE LEFT TO THE RIGHT");
          System.out.println("TOWER, ONE AT A TIME, NEVER PUTTING A LARGER DISK ON A");
          System.out.println("SMALLER DISK.\n");

          nextStep = Step.SELECT_TOTAL_DISKS;  // 设置下一步为选择总盘数
          break;  // 结束当前的 case，跳出 switch 语句

        case SELECT_TOTAL_DISKS:  // 当前的 case 语句为选择总的磁盘数

          while (numErrors <= 2) {  // 当错误次数小于等于2时执行循环

            // 获取用户输入
            System.out.print("HOW MANY DISKS DO YOU WANT TO MOVE (" + MAX_NUM_ROWS + " IS MAX)? ");
            numDisks = scan.nextInt();  // 从用户输入中获取磁盘数
            System.out.println("");

            numMoves = 0;  // 将移动次数重置为0

            // 确保磁盘数是有效的
            if ((numDisks < 1) || (numDisks > MAX_NUM_ROWS)) {  // 如果磁盘数小于1或大于最大行数

              numErrors++;  // 错误次数加1

              // 处理用户输入错误
          if (numErrors < 3) {  // 如果用户输入错误次数小于3次
            System.out.println("SORRY, BUT I CAN'T DO THAT JOB FOR YOU.");  // 打印提示信息
          }
          else {  // 否则
            break;  // 退出 while 循环
          }

          // 如果用户输入错误次数超过2次
          if (numErrors > 2) {
            System.out.println("ALL RIGHT, WISE GUY, IF YOU CAN'T PLAY THE GAME RIGHT, I'LL");  // 打印提示信息
            System.out.println("JUST TAKE MY PUZZLE AND GO HOME.  SO LONG.");  // 打印提示信息
            return;  // 结束方法的执行
          }

          // 显示详细的说明
          System.out.println("IN THIS PROGRAM, WE SHALL REFER TO DISKS BY NUMERICAL CODE.");  // 打印提示信息
          System.out.println("3 WILL REPRESENT THE SMALLEST DISK, 5 THE NEXT SIZE,");  // 打印提示信息
          // 打印游戏规则和提示信息
          System.out.println("7 THE NEXT, AND SO ON, UP TO 15.  IF YOU DO THE PUZZLE WITH");
          System.out.println("2 DISKS, THEIR CODE NAMES WOULD BE 13 AND 15.  WITH 3 DISKS");
          System.out.println("THE CODE NAMES WOULD BE 11, 13 AND 15, ETC.  THE NEEDLES");
          System.out.println("ARE NUMBERED FROM LEFT TO RIGHT, 1 TO 3.  WE WILL");
          System.out.println("START WITH THE DISKS ON NEEDLE 1, AND ATTEMPT TO MOVE THEM");
          System.out.println("TO NEEDLE 3.\n");
          System.out.println("GOOD LUCK!\n");

          // 初始化最大大小的盘子
          disk = MAX_DISK_SIZE;

          // 设置盘子的起始位置
          for (row = MAX_NUM_ROWS; row > (MAX_NUM_ROWS - numDisks); row--) {
            positions[row][1] = disk;
            disk = disk - 2;
          }

          // 打印盘子的起始位置
          printPositions();

          // 设置下一步为选择盘子移动
          nextStep = Step.SELECT_DISK_MOVE;
          break;
        case SELECT_DISK_MOVE:  # 选择移动磁盘的操作

          System.out.print("WHICH DISK WOULD YOU LIKE TO MOVE? ");  # 打印提示信息，询问用户想要移动哪个磁盘

          numErrors = 0;  # 初始化错误计数器

          while (numErrors < 2) {  # 进入循环，当错误计数器小于2时执行以下操作
            disk = scan.nextInt();  # 从用户输入中获取磁盘编号

            // Validate disk numbers  # 验证磁盘编号是否合法
            if ((disk - 3) * (disk - 5) * (disk - 7) * (disk - 9) * (disk - 11) * (disk - 13) * (disk - 15) == 0) {  # 检查磁盘编号是否在3、5、7、9、11、13、15中

              // Check if disk exists  # 检查磁盘是否存在
              diskPosition = getDiskPosition(disk);  # 获取磁盘的位置信息

              // Disk found  # 如果找到了磁盘
              if ((diskPosition.row > 0) && (diskPosition.column > 0))  # 如果磁盘的行和列都大于0
              {
// 如果磁盘可以移动
if (isDiskMovable(disk, diskPosition.row, diskPosition.column) == true) {
  // 跳出循环
  break;
}
// 如果磁盘不能移动
else {
  // 打印提示信息
  System.out.println("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.");
  System.out.print("WHICH DISK WOULD YOU LIKE TO MOVE? ");
}
// 模拟旧版本处理有效磁盘编号但未找到磁盘的情况
else {
  // 打印提示信息
  System.out.println("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.");
  System.out.print("WHICH DISK WOULD YOU LIKE TO MOVE? ");
  // 重置错误次数
  numErrors = 0;
}
                continue;  // 继续下一次循环

              }

            }
            // Invalid disk number  // 无效的磁盘号
            else {

              System.out.println("ILLEGAL ENTRY... YOU MAY ONLY TYPE 3,5,7,9,11,13, OR 15.");  // 输出错误信息
              numErrors++;  // 错误计数加一

              if (numErrors > 1) {  // 如果错误计数大于1
                break;  // 退出循环
              }

              System.out.print("? ");  // 输出提示符

            }
          }
          if (numErrors > 1) {  # 如果错误次数大于1
            System.out.println("STOP WASTING MY TIME.  GO BOTHER SOMEONE ELSE.");  # 输出错误信息
            return;  # 结束程序
          }

          nextStep = Step.SELECT_NEEDLE;  # 设置下一步操作为选择针
          break;  # 跳出循环


        case SELECT_NEEDLE:  # 选择针的情况

          numErrors = 0;  # 错误次数重置为0

          while (true) {  # 进入循环

            System.out.print("PLACE DISK ON WHICH NEEDLE? ");  # 输出提示信息
            needle = scan.nextInt();  # 从输入中获取针的位置

            // Handle valid needle numbers  # 处理有效的针号
            // 检查 needle 是否为0，1，2，3中的一个，如果是则进入条件判断
            if ((needle - 1) * (needle - 2) * (needle - 3) == 0) {

              // 确保 needle 对于磁盘移动是安全的
              if (isNeedleSafe(needle, disk, row) == false) {
                // 打印警告信息
                System.out.println("YOU CAN'T PLACE A LARGER DISK ON TOP OF A SMALLER ONE,");
                System.out.println("IT MIGHT CRUSH IT!");
                System.out.print("NOW THEN, ");
                // 设置下一步操作为选择磁盘移动
                nextStep = Step.SELECT_DISK_MOVE;
                // 跳出循环
                break;
              }

              // 获取磁盘的位置
              diskPosition = getDiskPosition(disk);

              // 尝试在非空的 needle 上移动磁盘
              diskMoved = false;
              for (row = 1; row <= MAX_NUM_ROWS; row++) {
                // 如果该位置上有磁盘
                if (positions[row][needle] != 0) {
                  // 将 row 减一
                  row--;
                  positions[row][needle] = positions[diskPosition.row][diskPosition.column];
                  // 将当前位置的磁盘移动到指定的 needle 上

                  positions[diskPosition.row][diskPosition.column] = 0;
                  // 将当前位置的磁盘移出原位置

                  diskMoved = true;
                  // 标记磁盘已经移动
                  break;
                }
              }

              // Needle was empty, so move disk to the bottom
              // 如果 needle 是空的，将磁盘移动到底部
              if (diskMoved == false) {
                positions[MAX_NUM_ROWS][needle] = positions[diskPosition.row][diskPosition.column];
                // 将磁盘移动到指定的 needle 的底部
                positions[diskPosition.row][diskPosition.column] = 0;
                // 将当前位置的磁盘移出原位置
              }

              nextStep = Step.CHECK_SOLUTION;
              // 设置下一步为检查解决方案
              break;

            }
            // Handle invalid needle numbers
            // 处理无效的 needle 编号
            else {
              # 如果输入的不是数字，则错误计数加一
              numErrors++;
              # 如果错误计数大于1，则输出警告信息并结束程序
              if (numErrors > 1) {
                System.out.println("I TRIED TO WARN YOU, BUT YOU WOULDN'T LISTEN.");
                System.out.println("BYE BYE, BIG SHOT.");
                return;
              }
              # 如果错误计数不大于1，则输出提示信息
              else {
                System.out.println("I'LL ASSUME YOU HIT THE WRONG KEY THIS TIME.  BUT WATCH IT,");
                System.out.println("I ONLY ALLOW ONE MISTAKE.");
              }
            }
          }
          # 跳出循环
          break;
        case CHECK_SOLUTION:  // 检查解决方案的情况

          printPositions();  // 打印当前位置信息

          numMoves++;  // 移动次数加一

          // Puzzle is solved  // 拼图已解决
          if (isPuzzleSolved() == true) {  // 如果拼图已解决

            // Check for optimal solution  // 检查最佳解决方案
            if (numMoves == (Math.pow(2, numDisks) - 1)) {  // 如果移动次数等于（2的numDisks次方减1）

              System.out.println("");  // 打印空行
              System.out.println("CONGRATULATIONS!!\n");  // 打印祝贺信息

            }

            System.out.println("YOU HAVE PERFORMED THE TASK IN " + numMoves + " MOVES.\n");  // 打印完成任务所需的移动次数
            System.out.print("TRY AGAIN (YES OR NO)? ");  // 提示重新尝试（是或否）？

            // Prompt for retries  // 提示重新尝试
            while (true) {  // 循环直到条件不满足
              userResponse = scan.next();  // 从用户输入中读取下一行字符串

              if (userResponse.toUpperCase().equals("YES")) {  // 如果用户输入转换为大写后等于"YES"
                nextStep = Step.INITIALIZE;  // 设置下一步操作为初始化
                break;  // 跳出循环
              }
              else if (userResponse.toUpperCase().equals("NO")) {  // 如果用户输入转换为大写后等于"NO"
                System.out.println("");  // 输出空行
                System.out.println("THANKS FOR THE GAME!\n");  // 输出感谢信息
                return;  // 结束程序
              }
              else {  // 如果用户输入既不是"YES"也不是"NO"
                System.out.print("'YES' OR 'NO' PLEASE? ");  // 输出提示信息
              }
            }
          }
          // Puzzle is not solved  // 拼图未解决
          else {

            // Exceeded maximum number of moves  // 超过最大移动次数
            if (numMoves > MAX_NUM_MOVES) {  // 如果移动次数超过最大限制
              System.out.println("SORRY, BUT I HAVE ORDERS TO STOP IF YOU MAKE MORE THAN");  // 打印提示信息
              System.out.println("128 MOVES.");  // 打印提示信息
              return;  // 结束程序
            }

            nextStep = Step.SELECT_DISK_MOVE;  // 设置下一步操作为选择磁盘移动
            break;  // 跳出当前循环
          }

          break;  // 跳出当前循环

        default:  // 默认情况
          System.out.println("INVALID STEP");  // 打印提示信息
          break;  // 跳出当前循环

      }

    }  // 结束外部循环
  }  // End of method startGame  // 方法startGame结束

  private boolean isPuzzleSolved() {  // 定义一个私有方法isPuzzleSolved，返回布尔值

    int column = 0;  // 初始化列变量为0
    int row = 0;  // 初始化行变量为0

    // Puzzle is solved if first 2 needles are empty  // 如果前两个针是空的，那么谜题已解决
    for (row = 1; row <= MAX_NUM_ROWS; row++) {  // 遍历行，从1到最大行数
      for (column = 1; column <= 2; column++) {  // 遍历列，从1到2
        if (positions[row][column] != 0) {  // 如果位置上的值不为0
          return false;  // 返回false
        }
      }
    }

    return true;  // 返回true，表示谜题已解决

  }  // End of method isPuzzleSolved  // 方法isPuzzleSolved结束
    # 根据磁盘号获取磁盘的位置
    private Position getDiskPosition(int disk) {
    
        int column = 0;
        int row = 0;
    
        Position pos = new Position(0, 0);
    
        # 开始循环遍历所有行
        for (row = 1; row <= MAX_NUM_ROWS; row++) {
    
            # 开始循环遍历所有列
            for (column = 1; column <= MAX_NUM_COLUMNS; column++) {
    
                # 找到了磁盘
                if (positions[row][column] == disk) {
    
                    pos.row = row;
                    pos.column = column;
          return pos;  // 返回当前位置

        }

      }  // 结束对所有列的循环

    }  // 结束对所有行的循环

    return pos;  // 返回当前位置

  }  // 结束 getDiskPosition 方法


  private boolean isDiskMovable(int disk, int row, int column) {

    int ii = 0;  // 循环迭代器

    // 开始循环遍历所有在磁盘上方的行
    for (ii = row; ii >= 1; ii--) {
      // 如果磁盘可以移动
      if (positions[ii][column] == 0) {
        continue;  // 继续下一次循环
      }

      // 如果磁盘不能移动
      if (positions[ii][column] < disk) {
        return false;  // 返回false
      }

    }  // 结束对磁盘上方所有行的循环

    return true;  // 返回true

  }  // 结束isDiskMovable方法


  private boolean isNeedleSafe(int needle, int disk, int row) {

    for (row = 1; row <= MAX_NUM_ROWS; row++) {
      // 如果 needle 不为空
      if (positions[row][needle] != 0) {
        // 磁盘压碎条件
        if (disk >= positions[row][needle]) {
          return false;
        }
      }
    }

    return true;

  }  // isNeedleSafe 方法结束


  private void printPositions() {

    int column = 1;
    int ii = 0;  // 循环迭代器
    int numSpaces = 0;  // 初始化变量numSpaces，用于表示空格数量
    int row = 1;  // 初始化变量row，表示当前行数

    // 开始循环遍历所有行
    for (row = 1; row <= MAX_NUM_ROWS; row++) {

      numSpaces = 9;  // 每行开始时，设置初始空格数量为9个

      // 开始循环遍历所有列
      for (column = 1; column <= MAX_NUM_COLUMNS; column++) {

        // 当前位置没有磁盘
        if (positions[row][column] == 0) {

          System.out.print(" ".repeat(numSpaces) + "*");  // 打印指定数量的空格和一个星号
          numSpaces = 20;  // 更新空格数量为20个
        }

        // 在当前位置绘制一个磁盘
        else {
          // 打印空格，使得星号图案居中显示
          System.out.print(" ".repeat(numSpaces - ((int) (positions[row][column] / 2)));

          // 打印星号，根据positions数组中的值确定打印的数量
          for (ii = 1; ii <= positions[row][column]; ii++) {
            System.out.print("*");
          }

          // 更新空格数量，使得下一行星号图案能够居中显示
          numSpaces = 20 - ((int) (positions[row][column] / 2));
        }

      }  // 结束对所有列的循环

      // 换行
      System.out.println("");

    }  // 结束对所有行的循环

  }  // 结束printPositions方法


  public static void main(String[] args) {
    Tower tower = new Tower();  // 创建一个名为tower的Tower对象
    tower.play();  // 调用Tower对象的play方法

  }  // End of method main  // main方法结束

}  // End of class Tower  // Tower类结束
```