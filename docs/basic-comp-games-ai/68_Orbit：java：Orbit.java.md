# `d:/src/tocomm/basic-computer-games\68_Orbit\java\Orbit.java`

```
import java.lang.Integer;  # 导入 Integer 类，用于处理整数
import java.lang.Math;  # 导入 Math 类，用于数学运算
import java.util.Scanner;  # 导入 Scanner 类，用于用户输入

/**
 * Game of Orbit
 * <p>
 * Based on the BASIC game of Orbit here
 * https://github.com/coding-horror/basic-computer-games/blob/main/68%20Orbit/orbit.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Orbit {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象
  public Orbit() {
    // 创建一个新的 Scanner 对象，用于从标准输入读取数据
    scan = new Scanner(System.in);
  }  // End of constructor Orbit

  public void play() {
    // 显示游戏介绍
    showIntro();
    // 开始游戏
    startGame();
  }  // End of method play

  private static void showIntro() {
    // 打印游戏标题
    System.out.println(" ".repeat(32) + "ORBIT");
    // 打印游戏信息
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

    // 打印游戏背景故事
    System.out.println("SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.");
    # 打印空行
    System.out.println("")
    # 打印提示信息
    System.out.println("THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS")
    System.out.println("DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM")
    System.out.println("10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN")
    System.out.println("CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.")
    System.out.println("")
    System.out.println("UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO")
    System.out.println("YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL")
    System.out.println("INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR")
    System.out.println("PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY")
    System.out.println("HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE")
    System.out.println("YOUR PLANET'S GRAVITY.")
    System.out.println("")
    System.out.println("YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.")
    System.out.println("")
    System.out.println("AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN")
    System.out.println("ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF")
    System.out.println("100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S")
    System.out.println("DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.")
    System.out.println("")
    # 打印出一段文本
    System.out.println("AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP");
    # 打印出一段文本
    System.out.println("WILL DESTROY IT.");
    # 打印出空行
    System.out.println("");
    # 打印出一段文本
    System.out.println("BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.");
    # 打印出空行
    System.out.println("");
    # 打印出空行
    System.out.println("");
    # 打印出一段文本
    System.out.println("                          90");
    # 打印出一段文本
    System.out.println("                    0000000000000");
    # 打印出一段文本
    System.out.println("                 0000000000000000000");
    # 打印出一段文本
    System.out.println("               000000           000000");
    # 打印出一段文本
    System.out.println("             00000                 00000");
    # 打印出一段文本
    System.out.println("            00000    XXXXXXXXXXX    00000");
    # 打印出一段文本
    System.out.println("           00000    XXXXXXXXXXXXX    00000");
    # 打印出一段文本
    System.out.println("          0000     XXXXXXXXXXXXXXX     0000");
    # 打印出一段文本
    System.out.println("         0000     XXXXXXXXXXXXXXXXX     0000");
    # 打印出一段文本
    System.out.println("        0000     XXXXXXXXXXXXXXXXXXX     0000");
    # 打印出一段文本
    System.out.println("180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0");
    # 打印出一段文本
    System.out.println("        0000     XXXXXXXXXXXXXXXXXXX     0000");
    # 打印出一段文本
    System.out.println("         0000     XXXXXXXXXXXXXXXXX     0000");
    # 打印出一段文本
    System.out.println("          0000     XXXXXXXXXXXXXXX     0000");
    # 打印出一段 ASCII 艺术，用于展示游戏的介绍
    System.out.println("           00000    XXXXXXXXXXXXX    00000");
    System.out.println("            00000    XXXXXXXXXXX    00000");
    System.out.println("             00000                 00000");
    System.out.println("               000000           000000");
    System.out.println("                 0000000000000000000");
    System.out.println("                    0000000000000");
    System.out.println("                         270");
    System.out.println("");
    System.out.println("X - YOUR PLANET");
    System.out.println("O - THE ORBIT OF THE ROMULAN SHIP");
    System.out.println("");
    System.out.println("ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING");
    System.out.println("COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT");
    System.out.println("WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE");
    System.out.println("AND ORBITAL RATE WILL REMAIN CONSTANT.");
    System.out.println("");
    System.out.println("GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.");
  
  }  # End of method showIntro
    private void startGame() {

        double bombDistance = 0; // 初始化炸弹距离
        int bombAltitude = 0; // 初始化炸弹高度
        int bombAngle = 0; // 初始化炸弹角度
        int deltaAngle = 0; // 初始化角度变化
        int hour = 0; // 初始化小时
        int shipAltitude = 0; // 初始化飞船高度
        int shipAngle = 0; // 初始化飞船角度
        int shipRate = 0; // 初始化飞船速率
        String userResponse = ""; // 初始化用户响应

        // 开始外部循环
        while (true) {
            shipAngle = (int) (361 * Math.random()); // 生成随机飞船角度
            shipAltitude = (int) (201 * Math.random() + 200); // 生成随机飞船高度
            shipRate = (int) (21 * Math.random() + 10); // 生成随机飞船速率

            hour = 0; // 重置小时为0
      // 开始时间限制循环
      while (hour < 7) {

        System.out.println("");
        System.out.println("");
        System.out.println("THIS IS HOUR " + (hour + 1) + ", AT WHAT ANGLE DO YOU WISH TO SEND");
        System.out.print("YOUR PHOTON BOMB? ");
        bombAngle = Integer.parseInt(scan.nextLine()); // 从用户输入中获取光子炸弹的角度

        System.out.print("HOW FAR OUT DO YOU WISH TO DETONATE IT? ");
        bombAltitude = Integer.parseInt(scan.nextLine()); // 从用户输入中获取炸弹的爆炸高度

        System.out.println("");
        System.out.println("");

        // 更新飞船位置
        shipAngle += shipRate; // 根据飞船速率更新飞船角度

        // 处理完整的旋转
        if (shipAngle >= 360) { // 如果飞船角度超过360度
          shipAngle -= 360;  // 将船的角度减去360度

        }

        deltaAngle = Math.abs(shipAngle - bombAngle);  // 计算船和炸弹之间的角度差的绝对值

        // 保持角度在上半象限
        if (deltaAngle >= 180) {  // 如果角度差大于等于180度
          deltaAngle = 360 - deltaAngle;  // 则将角度差更新为360度减去角度差
        }

        bombDistance = Math.sqrt(shipAltitude * shipAltitude + bombAltitude * bombAltitude - 2 * shipAltitude
                       * bombAltitude * Math.cos(deltaAngle * Math.PI / 180));  // 计算炸弹与船之间的距离

        System.out.format("YOUR PHOTON BOMB EXPLODED " + "%.5f" + "*10^2 MILES FROM THE\n", bombDistance);  // 打印炸弹爆炸的距离
        System.out.println("ROMULAN SHIP.");  // 打印罗穆兰飞船

        // 胜利条件
        if (bombDistance <= 50) {  // 如果炸弹距离小于等于50英里
          System.out.println("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.");  // 打印成功完成任务
          break;  // 退出循环
      }

      hour++;  // 增加小时计数

    }  // 结束时间限制循环

    // 失败条件
    if (hour == 7) {
      System.out.println("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.");
    }

    System.out.println("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.");  // 输出信息
    System.out.print("DO YOU WISH TO TRY TO DESTROY IT? ");  // 提示用户输入是否想要摧毁飞船
    userResponse = scan.nextLine();  // 读取用户输入

    if (!userResponse.toUpperCase().equals("YES")) {  // 如果用户输入不是"YES"
      System.out.println("GOOD BYE.");  // 输出信息
      break;  // 跳出循环
    }
    }  // 结束外部 while 循环

  }  // startGame 方法结束

  public static void main(String[] args) {

    Orbit game = new Orbit();
    game.play();

  }  // main 方法结束

}  // Orbit 类结束
```