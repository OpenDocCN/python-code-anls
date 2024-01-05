# `61_Math_Dice\java\Die.java`

```
import java.util.Random;  # 导入 Random 类

public class Die {  # 定义 Die 类
    private static final int DEFAULT_SIDES = 6;  # 定义默认的骰子面数为 6
    private int faceValue;  # 定义骰子的当前面值
    private int sides;  # 定义骰子的面数
    private Random generator = new Random();  # 创建一个 Random 对象用于生成随机数

    /**
     * Construct a new Die with default sides
     */
    public Die() {  # Die 类的构造函数
        this.sides = DEFAULT_SIDES;  # 将骰子的面数设为默认面数
        this.faceValue = 1 + generator.nextInt(sides);  # 生成一个随机数作为骰子的初始面值
    }

    /**
     * Generate a new random number between 1 and sides to be stored in faceValue
     */
    private void throwDie() {  # throwDie 方法用于生成一个新的随机数作为骰子的面值
        this.faceValue = 1 + generator.nextInt(sides);  # 生成一个1到骰子面数之间的随机数，并将其赋值给骰子的当前面值

    }

    /**
     * @return the faceValue  # 返回当前骰子的面值
     */
    public int getFaceValue() {
        return faceValue;
    }

    public void printDie() {
        throwDie();  # 掷骰子，生成一个随机的面值
        int x = this.getFaceValue();  # 获取当前骰子的面值

        System.out.println(" ----- ");  # 打印分隔线

        if(x==4||x==5||x==6) {  # 如果骰子的面值为4、5或6
            printTwo();  # 打印数字2的图案
        } else if(x==2||x==3) {  # 如果 x 等于 2 或者等于 3
            System.out.println("| *   |");  # 打印 "| *   |"
        } else {  # 否则
            printZero();  # 调用 printZero() 函数
        }

        if(x==1||x==3||x==5) {  # 如果 x 等于 1 或者等于 3 或者等于 5
            System.out.println("|  *  |");  # 打印 "|  *  |"
        } else if(x==2||x==4) {  # 否则如果 x 等于 2 或者等于 4
            printZero();  # 调用 printZero() 函数
        } else {  # 否则
            printTwo();  # 调用 printTwo() 函数
        }

        if(x==4||x==5||x==6) {  # 如果 x 等于 4 或者等于 5 或者等于 6
            printTwo();  # 调用 printTwo() 函数
        } else if(x==2||x==3) {  # 否则如果 x 等于 2 或者等于 3
            System.out.println("|   * |");  # 打印 "|   * |"
        } else {  # 否则
            printZero();  # 调用 printZero() 函数
        }  # 结束 printTwo 方法的定义

        System.out.println(" ----- ");  # 打印一行横线
    }

    private void printZero() {  # 定义一个名为 printZero 的私有方法
        System.out.println("|     |");  # 打印一个空格
    }

    private void printTwo() {  # 定义一个名为 printTwo 的私有方法
        System.out.println("| * * |");  # 打印一个带有星号的图案
    }
}
```