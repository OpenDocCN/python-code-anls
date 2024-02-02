# `basic-computer-games\45_Hello\javascript\hello.js`

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
                       // 设置输入框类型为文本，长度为50
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主控制部分，使用 async 关键字定义一个异步函数
async function main()
{
    // 打印 HELLO
    print(tab(33) + "HELLO\n");
    // 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    // 打印 HELLO.  MY NAME IS CREATIVE COMPUTER.
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n");
    // 打印两个空行
    print("\n");
    print("\n");
    // 打印 WHAT'S YOUR NAME
    print("WHAT'S YOUR NAME");
    // 调用输入函数，等待用户输入
    ns = await input();
    // 打印一个空行
    print("\n");
    // 打印 HI THERE, 用户输入的名字, ARE YOU ENJOYING YOURSELF HERE
    print("HI THERE, " + ns + ", ARE YOU ENJOYING YOURSELF HERE");
}
    # 进入循环，一直执行直到条件不满足
    while (1) {
        # 从输入中获取数据，赋值给变量 bs
        bs = await input();
        # 打印换行
        print("\n");
        # 如果输入为 "YES"，则执行以下代码块
        if (bs == "YES") {
            # 打印问候语和输入的名字
            print("I'M GLAD TO HEAR THAT, " + ns + ".\n");
            # 打印换行
            print("\n");
            # 跳出循环
            break;
        # 如果输入为 "NO"，则执行以下代码块
        } else if (bs == "NO") {
            # 打印安慰语和输入的名字
            print("OH, I'M SORRY TO HEAR THAT, " + ns + ". MAYBE WE CAN\n");
            print("BRIGHTEN UP YOUR VISIT A BIT.\n");
            # 跳出循环
            break;
        # 如果输入既不是 "YES" 也不是 "NO"，则执行以下代码块
        } else {
            # 提示用户输入 "YES" 或 "NO"
            print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE");
        }
    }
    # 打印换行
    print("\n");
    # 打印问候语和输入的名字，询问用户有什么问题
    print("SAY, " + ns + ", I CAN SOLVED ALL KINDS OF PROBLEMS EXCEPT\n");
    print("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO\n");
    print("YOU HAVE (ANSWER SEX, HEALTH, MONEY, OR JOB)");
// 打印两个空行
print("\n");
print("\n");
// 进入无限循环
while (1) {
    // 打印提示信息
    print("DID YOU LEAVE THE MONEY");
    // 等待用户输入
    gs = await input();
    // 打印空行
    print("\n");
    // 如果用户输入为"YES"
    if (gs == "YES") {
        // 打印相关信息
        print("HEY, " + ns + "??? YOU LEFT NO MONEY AT ALL!\n");
        print("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.\n");
        print("\n");
        print("WHAT A RIP OFF, " + ns + "!!!\n");
        print("\n");
        // 退出循环
        break;
    } 
    // 如果用户输入为"NO"
    else if (gs == "NO") {
        // 打印相关信息
        print("THAT'S HONEST, " + ns + ", BUT HOW DO YOU EXPECT\n");
        print("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENT\n");
        print("DON'T PAY THEIR BILLS?\n");
        // 退出循环
        break;
    } 
    // 如果用户输入不是"YES"也不是"NO"
    else {
        // 打印相关信息
        print("YOUR ANSWER OF '" + gs + "' CONFUSES ME, " + ns + ".\n");
        print("PLEASE RESPOND WITH 'YES' OR 'NO'.\n");
    }
}
// 打印两个空行
print("\n");
print("TAKE A WALK, " + ns + ".\n");
print("\n");
print("\n");
// 调用主函数
main();
```