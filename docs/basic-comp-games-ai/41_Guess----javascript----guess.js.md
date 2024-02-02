# `basic-computer-games\41_Guess\javascript\guess.js`

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
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个创建空行的函数
function make_space()
{
    for (h = 1; h <= 5; h++)
        // 打印换行符
        print("\n");
}

// 主控制部分，使用 async 函数定义
async function main()
{
    # 进入无限循环，直到条件为假
    while (1) {
        # 打印固定格式的字符串
        print(tab(33) + "GUESS\n");
        # 打印固定格式的字符串
        print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
        # 打印空行
        print("\n");
        print("\n");
        print("\n");
        # 打印固定格式的字符串
        print("THIS IS A NUMBER GUESSING GAME. I'LL THINK\n");
        print("OF A NUMBER BETWEEN 1 AND ANY LIMIT YOU WANT.\n");
        print("THEN YOU HAVE TO GUESS WHAT IT IS.\n");
        print("\n");

        # 打印固定格式的字符串
        print("WHAT LIMIT DO YOU WANT");
        # 从输入中获取整数并赋值给变量 l
        l = parseInt(await input());
        # 打印换行符
        print("\n");
        # 计算 l 的对数并赋值给变量 l1
        l1 = Math.floor(Math.log(l) / Math.log(2)) + 1;
        # 进入无限循环，直到条件为假
        while (1) {
            # 打印固定格式的字符串
            print("I'M THINKING OF A NUMBER BETWEEN 1 AND " + l + "\n");
            # 将变量 g 设为 1
            g = 1;
            # 打印固定格式的字符串
            print("NOW YOU TRY TO GUESS WHAT IT IS.\n");
            # 生成一个介于 1 和 l 之间的随机整数并赋值给变量 m
            m = Math.floor(l * Math.random() + 1);
            # 进入无限循环，直到条件为假
            while (1) {
                # 从输入中获取整数并赋值给变量 n
                n = parseInt(await input());
                # 如果 n 小于等于 0，则执行 make_space() 函数并跳出循环
                if (n <= 0) {
                    make_space();
                    break;
                }
                # 如果 n 等于 m，则打印固定格式的字符串并根据条件打印不同的消息，然后执行 make_space() 函数并跳出循环
                if (n == m) {
                    print("THAT'S IT! YOU GOT IT IN " + g + " TRIES.\n");
                    if (g == l1) {
                        print("GOOD.\n");
                    } else if (g < l1) {
                        print("VERY GOOD.\n");
                    } else {
                        print("YOU SHOULD HAVE BEEN TO GET IT IN ONLY " + l1 + "\n");
                    }
                    make_space();
                    break;
                }
                # 变量 g 自增 1
                g++;
                # 如果 n 大于 m，则打印固定格式的字符串；否则打印固定格式的字符串
                if (n > m)
                    print("TOO HIGH. TRY A SMALLER ANSWER.\n");
                else
                    print("TOO LOW. TRY A BIGGER ANSWER.\n");
            }
            # 如果 n 小于等于 0，则跳出循环
            if (n <= 0)
                break;
        }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```