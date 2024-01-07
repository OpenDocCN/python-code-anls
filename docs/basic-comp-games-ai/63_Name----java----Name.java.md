# `basic-computer-games\63_Name\java\Name.java`

```

# 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

# 定义一个名为 Name 的类
public class Name {

    # 定义一个静态方法，用于打印空行
    public static void printempty() { System.out.println(" "); }

    # 定义一个静态方法，用于打印字符串
    public static void print(String toprint) { System.out.println(toprint); }

    # 主方法
    public static void main(String[] args) {
        # 打印标题
        print("                                          NAME");
        print("                         CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        printempty();
        printempty();
        print("HELLO.");
        print("MY NAME iS CREATIVE COMPUTER.");
        print("WHATS YOUR NAME? (FIRST AND LAST)");

        # 创建一个 Scanner 对象，用于接收用户输入的名字
        Scanner namesc = new Scanner(System.in);
        String name = namesc.nextLine();

        # 将输入的名字进行反转
        String namereversed = new StringBuilder(name).reverse().toString();

        # 将输入的名字转换成字符数组，并进行排序
        char namesorted[] = name.toCharArray();
        Arrays.sort(namesorted);

        printempty();
        print("THANK YOU, " + namereversed);
        printempty();
        print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART");
        print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!");
        printempty();
        printempty();
        print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.");

        # 打印排序后的名字
        print("LET'S PUT THEM IN ORDER LIKE THIS: " + new String(namesorted));
        printempty();
        printempty();

        print("DON'T YOU LIKE THAT BETTER?");
        printempty();

        # 创建一个 Scanner 对象，用于接收用户对排序后名字的喜好
        Scanner agreementsc = new Scanner(System.in);
        String agreement = agreementsc.nextLine();

        # 根据用户的喜好进行不同的输出
        if (agreement.equalsIgnoreCase("yes")) {
            print("I KNEW YOU'D AGREE!!");
        } else {
            print("I'M SORRY YOU DON'T LIKE IT THAT WAY.");
            printempty();
            print("I REALLY ENJOYED MEETING YOU, " + name);
            print("HAVE A NICE DAY!");
        }
    }
}

```