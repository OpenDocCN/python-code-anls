# `basic-computer-games\30_Cube\java\src\Cube.java`

```
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

/**
 * Game of Cube
 * <p>
 * Based on game of Cube at:
 * https://github.com/coding-horror/basic-computer-games/blob/main/30_Cube/cube.bas
 *
 *
 */
public class Cube {

    //Current player location
    private Location playerLocation;

    //Current list of mines
    private Set<Location> mines;

    //System input / output objects
    private PrintStream out;
    private Scanner scanner;

    //Player's current money
    private int money;

    /**
     * Entry point, creates a new Cube object and calls the play method
     * @param args Java execution arguments, not used in application
     */
    public static void main(String[] args) {
        new Cube().play();
    }

    public Cube() {
        out = System.out;
        scanner = new Scanner(System.in);
        money = 500;
        mines = new HashSet<>(5);
    }

    /**
     * Clears mines and places 5 new mines on the board
     */
    private void placeMines() {
        mines.clear();
        Random random = new Random();
        for(int i = 0; i < 5; i++) {
            int x = random.nextInt(1,4);  // Generate a random integer between 1 and 4 for x coordinate
            int y = random.nextInt(1,4);  // Generate a random integer between 1 and 4 for y coordinate
            int z = random.nextInt(1,4);  // Generate a random integer between 1 and 4 for z coordinate
            mines.add(new Location(x,y,z));  // Add a new Location object with the random coordinates to the mines set
        }
    }

    /**
     * Runs the entire game until the player runs out of money or chooses to stop
     */
    }

    /**
     * Queries the user whether they want to play another round
     * @return True if the player decides to play another round,
     * False if the player would not like to play again
     */
    private boolean doAnotherRound() {
        if(money > 0) {
            out.println("DO YOU WANT TO TRY AGAIN?");  // Print a message asking the user if they want to play again
            return readParsedBoolean();  // Return the result of the readParsedBoolean method
        } else {
            return false;  // Return false if the player has no money left
        }
    }

    /**
     * Prints the instructions to the game, copied from the original code.
     */
    // 打印游戏说明
    public void printInstructions() {
        out.println("THIS IS A GAME IN WHICH YOU WILL BE PLAYING AGAINST THE");
        out.println("RANDOM DECISION OF THE COMPUTER. THE FIELD OF PLAY IS A");
        out.println("CUBE OF SIDE 3. ANY OF THE 27 LOCATIONS CAN BE DESIGNATED");
        out.println("BY INPUTTING THREE NUMBERS SUCH AS 2,3,1. AT THE START");
        out.println("YOU ARE AUTOMATICALLY AT LOCATION 1,1,1. THE OBJECT OF");
        out.println("THE GAME IS TO GET TO LOCATION 3,3,3. ONE MINOR DETAIL:");
        out.println("THE COMPUTER WILL PICK, AT RANDOM, 5 LOCATIONS AT WHICH");
        out.println("IT WILL PLANT LAND MINES. IF YOU HIT ONE OF THESE LOCATIONS");
        out.println("YOU LOSE. ONE OTHER DETAIL: YOU MAY MOVE ONLY ONE SPACE");
        out.println("IN ONE DIRECTION EACH MOVE. FOR  EXAMPLE: FROM 1,1,2 YOU");
        out.println("MAY MOVE TO 2,1,2 OR 1,1,3. YOU MAY NOT CHANGE");
        out.println("TWO OF THE NUMBERS ON THE SAME MOVE. IF YOU MAKE AN ILLEGAL");
        out.println("MOVE, YOU LOSE AND THE COMPUTER TAKES THE MONEY YOU MAY");
        out.println("\n");
        out.println("ALL YES OR NO QUESTIONS WILL BE ANSWERED BY A 1 FOR YES");
        out.println("OR A 0 (ZERO) FOR NO.");
        out.println();
        out.println("WHEN STATING THE AMOUNT OF A WAGER, PRINT ONLY THE NUMBER");
        out.println("OF DOLLARS (EXAMPLE: 250)  YOU ARE AUTOMATICALLY STARTED WITH");
        out.println("500 DOLLARS IN YOUR ACCOUNT.");
        out.println();
        out.println("GOOD LUCK!");
    }

    /**
     * 等待用户输入布尔值。可以是 (true,false), (1,0), (y,n), (yes,no) 等。
     * 默认情况下，返回 false
     * @return 用户输入的布尔值
     */
    // 读取用户输入的字符串，并尝试解析成布尔值
    private boolean readParsedBoolean() {
        String in = scanner.nextLine();
        try {
            // 将输入字符串转换为小写，并检查第一个字符是否为'y'，或者尝试解析成布尔值，或者解析成整数是否等于1
            return in.toLowerCase().charAt(0) == 'y' || Boolean.parseBoolean(in) || Integer.parseInt(in) == 1;
        } catch(NumberFormatException exception) {
            // 捕获异常，返回false
            return false;
        }
    }

    /**
     * 检查移动是否有效
     * @param from 玩家所在的位置
     * @param to 玩家希望移动到的位置
     * @return 如果玩家最多只能在任何方向上移动1个位置，则返回True，否则返回False
     */
    private boolean isMoveValid(Location from, Location to) {
        // 计算两个位置在三个坐标轴上的距离之和，判断是否小于等于1
        return Math.abs(from.x - to.x) + Math.abs(from.y - to.y) + Math.abs(from.z - to.z) <= 1;
    }

    public class Location {
        int x,y,z;

        public Location(int x, int y, int z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        /*
        用于在HashSet中使用，并检查两个位置是否相同
         */
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Location location = (Location) o;

            if (x != location.x) return false;
            if (y != location.y) return false;
            return z == location.z;
        }

        /*
        用于在HashSet中使用，相应地索引集合
         */
        @Override
        public int hashCode() {
            int result = x;
            result = 31 * result + y;
            result = 31 * result + z;
            return result;
        }
    }
# 闭合前面的函数定义
```