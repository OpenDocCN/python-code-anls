# `basic-computer-games\92_Trap\javascript\trap.js`

```py
// TRAP
//
// 由Oscar Toledo G. (nanochess)将BASIC转换为Javascript
//

// 打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数，返回一个Promise对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析Promise
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主控制部分
async function main()
{
    // 打印标题
    print(tab(34) + "TRAP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    g = 6;
    n = 100;
    // Trap
    // Steve Ullman, Aug/01/1972
    // 打印说明
    print("INSTRUCTIONS");
    // 等待输入
    str = await input();
}
    # 如果字符串的第一个字符是"Y"
    if (str.substr(0, 1) == "Y") {
        # 打印猜数字游戏的提示信息
        print("I AM THINKING OF A NUMBER BETWEEN 1 AND " + n + "\n");
        print("TRY TO GUESS MY NUMBER. ON EACH GUESS,\n");
        print("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP\n");
        print("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL\n");
        print("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY\n");
        print("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF\n");
        print("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.\n");
        print("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE\n");
        print("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.\n");
        print("YOU GET " + g + " GUESSES TO GET MY NUMBER.\n");
    }
    # 无限循环
    while (1) {
        # 生成一个1到n之间的随机整数
        x = Math.floor(n * Math.random()) + 1;
        # 循环进行猜数字游戏
        for (q = 1; q <= g; q++) {
            print("\n");
            print("GUESS #" + q + " ");
            # 等待用户输入两个数字
            str = await input();
            # 将输入的字符串转换为整数
            a = parseInt(str);
            # 获取逗号后的数字并转换为整数
            b = parseInt(str.substr(str.indexOf(",") + 1));
            # 如果用户猜中了
            if (a == b && x == a) {
                print("YOU GOT IT!!!\n");
                # 结束循环
                break;
            }
            # 如果a大于b，交换两个数字的值
            if (a > b) {
                r = a;
                a = b;
                b = r;
            }
            # 判断随机数x是否在用户输入的两个数字之间
            if (a <= x && x <= b) {
                print("YOU HAVE TRAPPED MY NUMBER.\n");
            } else if (x >= a) {
                print("MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.\n");
            } else {
                print("MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.\n");
            }
        }
        print("\n");
        print("TRY AGAIN.\n");
        print("\n");
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```