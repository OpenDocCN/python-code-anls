# `basic-computer-games\90_Tower\java\Tower.java`

```
import java.lang.Math;
import java.util.Scanner;

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

  private final static int MAX_DISK_SIZE = 15;

  private final static int MAX_NUM_COLUMNS = 3;

  private final static int MAX_NUM_MOVES = 128;

  private final static int MAX_NUM_ROWS = 7;

  private final Scanner scan;  // For user input

  // Represent all possible disk positions
  private int[][] positions;

  private enum Step {
    INITIALIZE, SELECT_TOTAL_DISKS, SELECT_DISK_MOVE, SELECT_NEEDLE, CHECK_SOLUTION
  }


  public Tower() {

    scan = new Scanner(System.in);

    // Row 0 and column 0 are not used
    positions = new int[MAX_NUM_ROWS + 1][MAX_NUM_COLUMNS + 1];

  }  // End of constructor Tower


  public class Position {

    public int row;
    public int column;

    public Position(int row, int column) {
      this.row = row;
      this.column = column;

    }  // End of constructor Position

  }  // End of inner class Position


  public void play() {

    showIntro();
    startGame();

  }  // End of method play


  private void showIntro() {

    System.out.println(" ".repeat(32) + "TOWERS");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro


  private void startGame() {

    boolean diskMoved = false;

    int column = 0;
    int disk = 0;
    int needle = 0;
    int numDisks = 0;
    int numErrors = 0;
    int numMoves = 0;
    int row = 0;

    Step nextStep = Step.INITIALIZE;

    String userResponse = "";

    Position diskPosition = new Position(0, 0);
    // 开始外部 while 循环
    }  // 结束外部 while 循环

  }  // startGame 方法结束


  private boolean isPuzzleSolved() {

    int column = 0;
    int row = 0;

    // 如果前两个针都是空的，那么谜题就解开了
    for (row = 1; row <= MAX_NUM_ROWS; row++) {
      for (column = 1; column <= 2; column++) {
        if (positions[row][column] != 0) {
          return false;
        }
      }
    }

    return true;

  }  // isPuzzleSolved 方法结束


  private Position getDiskPosition(int disk) {

    int column = 0;
    int row = 0;

    Position pos = new Position(0, 0);

    // 开始遍历所有行
    for (row = 1; row <= MAX_NUM_ROWS; row++) {

      // 开始遍历所有列
      for (column = 1; column <= MAX_NUM_COLUMNS; column++) {

        // 找到了磁盘
        if (positions[row][column] == disk) {

          pos.row = row;
          pos.column = column;
          return pos;

        }

      }  // 结束遍历所有列

    }  // 结束遍历所有行

    return pos;

  }  // getDiskPosition 方法结束


  private boolean isDiskMovable(int disk, int row, int column) {

    int ii = 0;  // 循环迭代器

    // 开始遍历磁盘上方的所有行
    for (ii = row; ii >= 1; ii--) {

      // 磁盘可以移动
      if (positions[ii][column] == 0) {
        continue;
      }

      // 磁盘无法移动
      if (positions[ii][column] < disk) {
        return false;
      }

    }  // 结束遍历磁盘上方的所有行

    return true;

  }  // isDiskMovable 方法结束


  private boolean isNeedleSafe(int needle, int disk, int row) {

    for (row = 1; row <= MAX_NUM_ROWS; row++) {

      // 针不是空的
      if (positions[row][needle] != 0) {

        // 磁盘压扁条件
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
    int numSpaces = 0;  // 空格数量
    int row = 1;  // 行数

    // 开始循环遍历所有行
    for (row = 1; row <= MAX_NUM_ROWS; row++) {

      numSpaces = 9;  // 初始空格数量

      // 开始循环遍历所有列
      for (column = 1; column <= MAX_NUM_COLUMNS; column++) {

        // 当前位置没有磁盘
        if (positions[row][column] == 0) {

          System.out.print(" ".repeat(numSpaces) + "*");  // 打印空格和星号
          numSpaces = 20;  // 更新空格数量
        }

        // 在当前位置绘制磁盘
        else {

          System.out.print(" ".repeat(numSpaces - ((int) (positions[row][column] / 2))));  // 打印空格
          
          for (ii = 1; ii <= positions[row][column]; ii++) {
            System.out.print("*");  // 打印磁盘
          }

          numSpaces = 20 - ((int) (positions[row][column] / 2));  // 更新空格数量
        }

      }  // 结束循环遍历所有列

      System.out.println("");  // 换行

    }  // 结束循环遍历所有行

  }  // 方法 printPositions 结束


  public static void main(String[] args) {

    Tower tower = new Tower();
    tower.play();  // 调用 play 方法

  }  // 方法 main 结束
}  // Tower 类的结束
```