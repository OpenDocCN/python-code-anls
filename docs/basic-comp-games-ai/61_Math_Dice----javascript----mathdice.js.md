# `basic-computer-games\61_Math_Dice\javascript\mathdice.js`

```
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
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析输入的值
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

// 主程序
async function main()
{
    // 打印标题
    print(tab(31) + "MATH DICE\n");
    // 打印副标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印多个空行
    print("\n");
    print("\n");
    print("\n");
    // 打印提示信息
    print("THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE.\n");
    print("WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION\n");
    print("MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY.\n");
}
    # 打印提示信息，要求在结束课程时将答案输入为零
    print("TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.\n");
    # 打印空行
    print("\n");
    print("\n");
    # 初始化变量 n 为 0
    n = 0;
    # 进入无限循环
    while (1) {
        # n 自增
        n++;
        # 生成一个 1 到 6 之间的随机整数
        d = Math.floor(6 * Math.random() + 1);
        # 打印分隔线
        print(" ----- \n");
        # 根据随机数 d 的值打印不同的骰子图案
        if (d == 1)
            print("I     I\n");
        else if (d == 2 || d == 3)
            print("I *   I\n");
        else
            print("I * * I\n");
        if (d == 2 || d == 4)
            print("I     I\n");
        else if (d == 6)
            print("I * * I\n");
        else
            print("I  *  I\n");
        if (d == 1)
            print("I     I\n");
        else if (d == 2 || d == 3)
            print("I   * I\n");
        else
            print("I * * I\n");
        # 打印分隔线
        print(" ----- \n");
        print("\n");
        # 如果 n 不等于 2，则继续循环
        if (n != 2) {
            # 打印加号
            print("   +\n");
            print("\n");
            # 将当前骰子点数赋值给变量 a，然后继续循环
            a = d;
            continue;
        }
        # 计算当前骰子点数和上一次的点数之和
        t = d + a;
        # 打印等号
        print("      =");
        # 将用户输入的值转换为整数
        t1 = parseInt(await input());
        # 如果用户输入为 0，则跳出循环
        if (t1 == 0)
            break;
        # 如果用户输入不等于计算得到的值 t，则进行以下判断
        if (t1 != t) {
            # 打印提示信息，要求重新计算点数并输入答案
            print("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.\n");
            print("      =");
            # 将用户输入的值转换为整数
            t1 = parseInt(await input());
            # 如果用户输入的值仍然不等于计算得到的值 t，则打印正确答案
            if (t1 != t) {
                print("NO, THE ANSWER IS " + t + "\n");
            }
        }
        # 如果用户输入的值等于计算得到的值 t，则打印正确提示
        if (t1 == t) {
            print("RIGHT!\n");
        }
        # 打印空行
        print("\n");
        # 打印提示信息，骰子重新投掷
        print("THE DICE ROLL AGAIN...\n");
        print("\n");
        # 重置变量 n 为 0，重新开始循环
        n = 0;
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```