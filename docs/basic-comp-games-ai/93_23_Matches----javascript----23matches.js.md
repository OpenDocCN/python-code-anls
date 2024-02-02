# `basic-computer-games\93_23_Matches\javascript\23matches.js`

```py
// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主控制部分，使用 async 函数定义
async function main()
{
    // 打印游戏标题
    print(tab(31) + "23 MATCHES\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print(" THIS IS A GAME CALLED '23 MATCHES'.\n");
    print("\n");
    print("WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE\n");
    print("MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE\n");
    print("THE LAST MATCH.\n");
}
    # 打印空行
    print("\n");
    # 打印提示信息
    print("LET'S FLIP A COIN TO SEE WHO GOES FIRST.\n");
    # 打印提示信息
    print("IF IT COMES UP HEADS, I WILL WIN THE TOSS.\n");
    # 打印空行
    print("\n");
    # 初始化变量 n
    n = 23;
    # 生成一个 0 或 1 的随机数，并向下取整
    q = Math.floor(2 * Math.random());
    # 如果随机数不为 1，则执行以下代码块
    if (q != 1) {
        # 打印提示信息
        print("TAILS! YOU GO FIRST. \n");
        # 打印空行
        print("\n");
    } else {
        # 打印提示信息
        print("HEADS! I WIN! HA! HA!\n");
        # 打印提示信息
        print("PREPARE TO LOSE, MEATBALL-NOSE!!\n");
        # 打印空行
        print("\n");
        # 打印提示信息
        print("I TAKE 2 MATCHES\n");
        # 更新变量 n
        n -= 2;
    }
    # 进入循环
    while (1) {
        # 如果 q 为 1，则执行以下代码块
        if (q == 1) {
            # 打印当前剩余的火柴数量
            print("THE NUMBER OF MATCHES IS NOW " + n + "\n");
            # 打印空行
            print("\n");
            # 打印提示信息
            print("YOUR TURN -- YOU MAY TAKE 1, 2 OR 3 MATCHES.\n");
        }
        # 打印提示信息
        print("HOW MANY DO YOU WISH TO REMOVE ");
        # 进入循环
        while (1) {
            # 将用户输入的值转换为整数
            k = parseInt(await input());
            # 如果输入值不在 1 到 3 之间，则执行以下代码块
            if (k <= 0 || k > 3) {
                # 打印提示信息
                print("VERY FUNNY! DUMMY!\n");
                # 打印提示信息
                print("DO YOU WANT TO PLAY OR GOOF AROUND?\n");
                # 打印提示信息
                print("NOW, HOW MANY MATCHES DO YOU WANT ");
            } else {
                # 跳出循环
                break;
            }
        }
        # 更新变量 n
        n -= k;
        # 打印剩余的火柴数量
        print("THERE ARE NOW " + n + " MATCHES REMAINING.\n");
        # 根据剩余的火柴数量更新变量 z
        if (n == 4) {
            z = 3;
        } else if (n == 3) {
            z = 2;
        } else if (n == 2) {
            z = 1;
        } else if (n > 1) {
            z = 4 - k;
        } else {
            # 打印提示信息
            print("YOU WON, FLOPPY EARS !\n");
            # 打印提示信息
            print("THINK YOU'RE PRETTY SMART !\n");
            # 打印提示信息
            print("LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!\n");
            # 跳出循环
            break;
        }
        # 打印提示信息
        print("MY TURN ! I REMOVE " + z + " MATCHES\n");
        # 更新变量 n
        n -= z;
        # 如果剩余的火柴数量小于等于 1，则执行以下代码块
        if (n <= 1) {
            # 打印提示信息
            print("\n");
            # 打印提示信息
            print("YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!\n");
            # 打印提示信息
            print("HA ! HA ! I BEAT YOU !!!\n");
            # 打印空行
            print("\n");
            # 打印提示信息
            print("GOOD BYE LOSER!\n");
            # 跳出循环
            break;
        }
        # 更新变量 q
        q = 1;
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```