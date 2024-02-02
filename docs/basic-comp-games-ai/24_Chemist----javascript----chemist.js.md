# `basic-computer-games\24_Chemist\javascript\chemist.js`

```py
// 定义一个打印函数，将字符串输出到指定的元素中
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

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并解析 Promise
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise
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

// 主程序
async function main()
{
    // 输出标题
    print(tab(33) + "CHEMIST\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 输出化学实验相关信息
    print("THE FICTITIOUS CHECMICAL KRYPTOCYANIC ACID CAN ONLY BE\n");
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.\n");
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE\n");
}
    # 打印警告信息，提醒用户即将发生的危险
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST\n");
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS\n");
    print("YOU FACE THE CONSEQUENCES.\n");
    # 初始化变量 t
    t = 0;
    # 进入循环，不断执行以下代码块
    while (1) {
        # 生成一个随机数 a，取整
        a = Math.floor(Math.random() * 50);
        # 根据 a 计算出需要的水量 w
        w = 7 * a / 3;
        # 打印提示信息，要求用户输入水的数量
        print(a + " LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER");
        # 获取用户输入的水量，转换为浮点数
        r = parseFloat(await input());
        # 计算用户输入水量与实际所需水量的差值
        d = Math.abs(w - r);
        # 如果差值超过实际所需水量的 5%，则执行以下代码块
        if (d > w / 20) {
            # 打印警告信息，用户失败，被转化为原生质
            print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB\n");
            print(" OF QUIVERING PROTOPLASM!\n");
            # t 自增 1
            t++;
            # 如果 t 达到 9，则跳出循环
            if (t == 9)
                break;
            # 打印提示信息，用户可以再次尝试
            print(" HOWEVER, YOU MAY TRY AGAIN WITH ANOTHER LIFE.\n");
        } else {
            # 打印祝贺信息，用户成功完成任务
            print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n");
            print("\n");
        }
    }
    # 打印最终结果信息，用户使用完 9 次机会
    print(" YOUR 9 LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR\n");
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```