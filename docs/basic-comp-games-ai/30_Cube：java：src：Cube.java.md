# `d:/src/tocomm/basic-computer-games\30_Cube\java\src\Cube.java`

```
import java.io.PrintStream;  # 导入打印流类
import java.util.HashSet;  # 导入哈希集合类
import java.util.Random;  # 导入随机数类
import java.util.Scanner;  # 导入扫描器类
import java.util.Set;  # 导入集合类

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
    private Location playerLocation;  # 当前玩家位置

    //Current list of mines
    private Set<Location> mines; // 创建一个名为mines的集合，用于存储地雷的位置

    //System input / output objects
    private PrintStream out; // 创建一个名为out的PrintStream对象，用于输出
    private Scanner scanner; // 创建一个名为scanner的Scanner对象，用于输入

    //Player's current money
    private int money; // 创建一个名为money的整数变量，用于存储玩家当前的金钱数量

    /**
     * Entry point, creates a new Cube object and calls the play method
     * @param args Java execution arguments, not used in application
     */
    public static void main(String[] args) {
        new Cube().play(); // 创建一个新的Cube对象并调用其play方法
    }

    public Cube() {
        out = System.out; // 将System.out赋值给out，用于输出
        scanner = new Scanner(System.in); // 创建一个新的Scanner对象，用于从控制台输入
        money = 500;  // 初始化玩家的初始金钱为500
        mines = new HashSet<>(5);  // 创建一个包含5个元素的空的HashSet集合
    }

    /**
     * Clears mines and places 5 new mines on the board
     */
    private void placeMines() {
        mines.clear();  // 清空之前的矿山数据
        Random random = new Random();  // 创建一个随机数生成器对象
        for(int i = 0; i < 5; i++) {  // 循环5次，放置5个新的矿山
            int x = random.nextInt(1,4);  // 生成一个1到4之间的随机整数作为x坐标
            int y = random.nextInt(1,4);  // 生成一个1到4之间的随机整数作为y坐标
            int z = random.nextInt(1,4);  // 生成一个1到4之间的随机整数作为z坐标
            mines.add(new Location(x,y,z));  // 将新生成的矿山坐标加入到矿山集合中
        }
    }

    /**
     * Runs the entire game until the player runs out of money or chooses to stop
    */
    public void play() {
        // 打印提示信息，询问是否需要查看游戏说明
        out.println("DO YOU WANT TO SEE INSTRUCTIONS? (YES--1,NO--0)");
        // 如果用户输入的是YES，则打印游戏说明
        if(readParsedBoolean()) {
            printInstructions();
        }
        // 放置地雷
        do {
            placeMines();
            // 询问是否要下注
            out.println("WANT TO MAKE A WAGER?");
            int wager = 0 ;
            // 如果用户选择下注
            if(readParsedBoolean()) {
                // 询问下注金额
                out.println("HOW MUCH?");
                // 循环直到输入合法的下注金额
                do {
                    wager = Integer.parseInt(scanner.nextLine());
                    // 如果下注金额大于玩家拥有的金额，则提示重新下注
                    if(wager > money) {
                        out.println("TRIED TO FOOL ME; BET AGAIN");
                    }
                } while(wager > money);
            }
            playerLocation = new Location(1,1,1);  # 初始化玩家位置为(1,1,1)
            while(playerLocation.x + playerLocation.y + playerLocation.z != 9) {  # 当玩家位置的坐标和不等于9时，进入循环
                out.println("\nNEXT MOVE");  # 输出提示信息
                String input = scanner.nextLine();  # 从控制台获取用户输入

                String[] stringValues = input.split(",");  # 将用户输入的字符串按逗号分割成数组

                if(stringValues.length < 3) {  # 如果输入的数组长度小于3
                    out.println("ILLEGAL MOVE, YOU LOSE.");  # 输出非法移动信息
                    return;  # 结束程序
                }

                int x = Integer.parseInt(stringValues[0]);  # 将字符串数组的第一个元素转换为整数
                int y = Integer.parseInt(stringValues[1]);  # 将字符串数组的第二个元素转换为整数
                int z = Integer.parseInt(stringValues[2]);  # 将字符串数组的第三个元素转换为整数

                Location location = new Location(x,y,z);  # 创建新的位置对象

                if(x < 1 || x > 3 || y < 1 || y > 3 || z < 1 || z > 3 || !isMoveValid(playerLocation,location)) {  # 如果新位置超出范围或者移动无效
                    out.println("ILLEGAL MOVE, YOU LOSE.");  // 输出提示信息，表示玩家走出了非法的移动，输掉了游戏
                    return;  // 结束当前方法的执行
                }

                playerLocation = location;  // 更新玩家的位置为新的位置

                if(mines.contains(location)) {  // 判断玩家当前位置是否包含地雷
                    out.println("******BANG******");  // 输出提示信息，表示玩家踩到了地雷
                    out.println("YOU LOSE!\n\n");  // 输出提示信息，表示玩家输掉了游戏
                    money -= wager;  // 玩家的金钱减去赌注
                    break;  // 跳出循环
                }
            }

            if(wager > 0) {  // 如果赌注大于0
                out.printf("YOU NOW HAVE %d DOLLARS\n",money);  // 输出提示信息，表示玩家当前的金钱数
            }

        } while(money > 0 && doAnotherRound());  // 当玩家的金钱数大于0并且需要进行另一轮游戏时继续循环
        out.println("TOUGH LUCK!");  # 打印“TOUGH LUCK!”到控制台
        out.println("\nGOODBYE.");  # 打印换行和“GOODBYE.”到控制台
    }

    /**
     * Queries the user whether they want to play another round
     * @return True if the player decides to play another round,
     * False if the player would not like to play again
     */
    private boolean doAnotherRound() {
        if(money > 0) {  # 如果玩家的钱大于0
            out.println("DO YOU WANT TO TRY AGAIN?");  # 打印“DO YOU WANT TO TRY AGAIN?”到控制台
            return readParsedBoolean();  # 调用readParsedBoolean()方法，返回用户输入的布尔值
        } else {
            return false;  # 如果玩家的钱不大于0，返回false
        }
    }

    /**
     * Prints the instructions to the game, copied from the original code.
    */
    # 打印游戏说明
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
        out.println("OF DOLLARS (EXAMPLE: 250)  YOU ARE AUTOMATICALLY STARTED WITH");  // 打印初始提示信息
        out.println("500 DOLLARS IN YOUR ACCOUNT.");  // 打印初始账户金额
        out.println();  // 打印空行
        out.println("GOOD LUCK!");  // 打印祝福信息
    }

    /**
     * 等待用户输入布尔值。可以是(true,false), (1,0), (y,n), (yes,no)等。
     * 默认情况下，返回false
     * @return 用户输入的布尔值
     */
    private boolean readParsedBoolean() {
        String in = scanner.nextLine();  // 读取用户输入的字符串
        try {
            return in.toLowerCase().charAt(0) == 'y' || Boolean.parseBoolean(in) || Integer.parseInt(in) == 1;  // 尝试解析用户输入的布尔值
        } catch(NumberFormatException exception) {
            return false;  // 捕获异常，返回false
        }
    }
    /**
     * 检查移动是否有效
     * @param from 玩家所在的位置
     * @param to 玩家希望移动到的位置
     * @return 如果玩家只能在任何方向上移动最多1个位置，则返回True，如果移动无效则返回False
     */
    private boolean isMoveValid(Location from, Location to) {
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
        // 用于 HashSet 和检查两个位置是否相同
         */
        @Override
        public boolean equals(Object o) {
            // 如果两个对象引用相同，则返回 true
            if (this == o) return true;
            // 如果 o 为 null 或者 o 的类与当前类不相同，则返回 false
            if (o == null || getClass() != o.getClass()) return false;

            // 将 o 转换为 Location 类型
            Location location = (Location) o;

            // 检查 x 坐标是否相同，如果不同则返回 false
            if (x != location.x) return false;
            // 检查 y 坐标是否相同，如果不同则返回 false
            if (y != location.y) return false;
            // 检查 z 坐标是否相同，如果不同则返回 false
            return z == location.z;
        }

        /*
        用于 HashSet 中相应地索引集合
         */
        @Override
        public int hashCode() {
            // 计算哈希码
            int result = x;
            result = 31 * result + y;  # 将 result 乘以 31，然后加上 y 的值
            result = 31 * result + z;  # 将 result 乘以 31，然后加上 z 的值
            return result;  # 返回最终的 result 值
        }
    }
}
```