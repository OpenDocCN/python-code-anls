# `basic-computer-games\69_Pizza\javascript\pizza.js`

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
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入框的值
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
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

// 定义一些数组
var sa = [, "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"];
var ma = [, "1","2","3","4"];
var a = [];

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "PIZZA\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("PIZZA DELIVERY GAME\n");
    print("\n");
    print("WHAT IS YOUR FIRST NAME");
    // 等待用户输入，并将输入的字符串赋值给变量 ns
    ns = await input();
    print("\n");
    # 打印欢迎词和游戏介绍
    print("HI, " + ns + ". IN THIS GAME YOU ARE TO TAKE ORDERS\n");
    print("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY\n");
    print("WHERE TO DELIVER THE ORDERED PIZZAS.\n");
    print("\n");
    print("\n");
    # 打印城市地图标题
    print("MAP OF THE CITY OF HYATTSVILLE\n");
    print("\n");
    # 打印城市地图的横线
    print(" -----1-----2-----3-----4-----\n");
    k = 4;
    # 循环打印城市地图的内容
    for (i = 1; i <= 4; i++) {
        print("-\n");
        print("-\n");
        print("-\n");
        print("-\n");
        # 打印城市地图每一行的内容
        print(ma[k]);
        s1 = 16 - 4 * i + 1;
        print("     " + sa[s1] + "     " + sa[s1 + 1] + "     " + sa[s1 + 2] + "     ");
        print(sa[s1 + 3] + "     " + ma[k] + "\n");
        k--;
    }
    # 打印城市地图的横线
    print("-\n");
    print("-\n");
    print("-\n");
    print("-\n");
    print(" -----1-----2-----3-----4-----\n");
    print("\n");
    # 打印送披萨的目的地介绍
    print("THE OUTPUT IS A MAP OF THE HOMES WHERE\n");
    print("YOU ARE TO SEND PIZZAS.\n");
    print("\n");
    print("YOUR JOB IS TO GIVE A TRUCK DRIVER\n");
    print("THE LOCATION OR COORDINATES OF THE\n");
    print("HOME ORDERING THE PIZZA.\n");
    print("\n");
    # 循环询问是否需要更多指示
    while (1) {
        print("DO YOU NEED MORE DIRECTIONS");
        str = await input();
        if (str == "YES" || str == "NO")
            break;
        print("'YES' OR 'NO' PLEASE, NOW THEN, ");
    }
    # 如果输入的字符串为"YES"，则执行以下操作
    if (str == "YES") {
        # 打印空行
        print("\n");
        # 打印提示信息
        print("SOMEBODY WILL ASK FOR A PIZZA TO BE\n");
        print("DELIVERED.  THEN A DELIVERY BOY WILL\n");
        print("ASK YOU FOR THE LOCATION.\n");
        print("     EXAMPLE:\n");
        print("THIS IS J.  PLEASE SEND A PIZZA.\n");
        print("DRIVER TO " + ns + ".  WHERE DOES J LIVE?\n");
        print("YOUR ANSWER WOULD BE 2,3\n");
        print("\n");
        print("UNDERSTAND");
        # 等待用户输入
        str = await input();
        # 如果输入不为"YES"，则打印提示信息并返回
        if (str != "YES") {
            print("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY");
            return;
        }
        # 打印提示信息
        print("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.\n");
        print("\n");
        print("GOOD LUCK!!\n");
        print("\n");
    }
    # 进入无限循环
    while (1) {
        # 循环5次
        for (i = 1; i <= 5; i++) {
            # 生成1到16之间的随机整数
            s = Math.floor(Math.random() * 16 + 1);
            # 打印提示信息
            print("\n");
            print("HELLO " + ns + "'S PIZZA.  THIS IS " + sa[s] + ".\n");
            print("  PLEASE SEND A PIZZA.\n");
            # 进入无限循环
            while (1) {
                # 打印提示信息并等待用户输入
                print("  DRIVER TO " + ns + ":  WHERE DOES " + sa[s] + " LIVE");
                str = await input();
                # 解析用户输入的字符串并赋值给数组a
                a[1] = parseInt(str);
                a[2] = parseInt(str.substr(str.indexOf(",") + 1));
                # 计算t的值
                t = a[1] + (a[2] - 1) * 4;
                # 如果t不等于s，则打印提示信息并继续循环
                if (t != s) {
                    print("THIS IS " + sa[t] + ". I DID NOT ORDER A PIZZA.\n");
                    print("I LIVE AT " + a[1] + "," + a[2] + "\n");
                } else {
                    # 否则跳出循环
                    break;
                }
            }
            # 打印提示信息
            print("HELLO " + ns + ".  THIS IS " + sa[s] + ", THANKS FOR THE PIZZA.\n");
        }
        # 打印提示信息并等待用户输入
        print("\n");
        print("DO YOU WANT TO DELIVER MORE PIZZAS");
        str = await input();
        # 如果输入不为"YES"，则跳出循环
        if (str != "YES")
            break;
    }
    # 打印提示信息
    print("\n");
    print("O.K. " + ns + ", SEE YOU LATER!\n");
    print("\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```