# `basic-computer-games\53_King\javascript\king.js`

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
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入的字符串
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

// 定义一个函数，打印一段特定的文本
function hate_your_guts()
{
    print("\n");
    print("\n");
    print("OVER ONE THIRD OF THE POPULATION HAS DIED SINCE YOU\n");
    print("WERE ELECTED TO OFFICE. THE PEOPLE (REMAINING)\n");
    print("HATE YOUR GUTS.\n");
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(34) + "KING\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
}
    # 打印提示信息，询问是否需要说明
    print("DO YOU WANT INSTRUCTIONS");
    # 获取用户输入的字符串
    str = await input();
    # 初始化变量 n5 为 8
    n5 = 8;
    # 如果用户输入的字符串为 "AGAIN"，则执行以下代码块
    if (str == "AGAIN") {
        # 进入循环，直到条件不满足
        while (1) {
            # 打印提示信息，询问被中断时你在任职多少年
            print("HOW MANY YEARS HAD YOU BEEN IN OFFICE WHEN INTERRUPTED");
            # 将用户输入的字符串转换为整数
            x5 = parseInt(await input());
            # 如果输入为 0，则返回
            if (x5 == 0)
                return;
            # 如果输入小于 8，则跳出循环
            if (x5 < 8)
                break;
            # 打印提示信息，提醒用户在任职年限
            print("   COME ON, YOUR TERM IN OFFICE IS ONLY " + n5 + " YEARS.\n");
        }
        # 打印提示信息，询问国库中有多少钱
        print("HOW MUCH DID YOU HAVE IN THE TREASURY");
        # 将用户输入的字符串转换为整数
        a = parseInt(await input());
        # 如果输入小于 0，则返回
        if (a < 0)
            return;
        # 打印提示信息，询问国家有多少人口
        print("HOW MANY COUNTRYMEN");
        # 将用户输入的字符串转换为整数
        b = parseInt(await input());
        # 如果输入小于 0，则返回
        if (b < 0)
            return;
        # 打印提示信息，询问有多少工人
        print("HOW MANY WORKERS");
        # 将用户输入的字符串转换为整数
        c = parseInt(await input());
        # 如果输入小于 0，则返回
        if (c < 0)
            return;
        # 进入循环，直到条件不满足
        while (1) {
            # 打印提示信息，询问有多少平方英里的土地
            print("HOW MANY SQUARE MILES OF LAND");
            # 将用户输入的字符串转换为整数
            d = parseInt(await input());
            # 如果输入小于 0，则返回
            if (d < 0)
                return;
            # 如果输入大于 1000且小于等于 2000，则跳出循环
            if (d > 1000 && d <= 2000)
                break;
            # 打印提示信息，提醒用户初始的土地面积
            print("   COME ON, YOU STARTED WITH 1000 SQ. MILES OF FARM LAND\n");
            print("   AND 10,000 SQ. MILES OF FOREST LAND.\n");
        }
    } else {
        # 如果条件不满足，则执行以下代码块
        if (str.substr(0, 1) != "N") {
            # 如果字符串的第一个字符不是"N"，则执行以下代码块
            print("\n");
            print("\n");
            print("\n");
            print("CONGRATULATIONS! YOU'VE JUST BEEN ELECTED PREMIER OF SETATS\n");
            print("DETINU, A SMALL COMMUNIST ISLAND 30 BY 70 MILES LONG. YOUR\n");
            print("JOB IS TO DECIDE UPON THE CONTRY'S BUDGET AND DISTRIBUTE\n");
            print("MONEY TO YOUR COUNTRYMEN FROM THE COMMUNAL TREASURY.\n");
            print("THE MONEY SYSTEM IS RALLODS, AND EACH PERSON NEEDS 100\n");
            print("RALLODS PER YEAR TO SURVIVE. YOUR COUNTRY'S INCOME COMES\n");
            print("FROM FARM PRODUCE AND TOURISTS VISITING YOUR MAGNIFICENT\n");
            print("FORESTS, HUNTING, FISHING, ETC. HALF YOUR LAND IS FARM LAND\n");
            print("WHICH ALSO HAS AN EXCELLENT MINERAL CONTENT AND MAY BE SOLD\n");
            print("TO FOREIGN INDUSTRY (STRIP MINING) WHO IMPORT AND SUPPORT\n");
            print("THEIR OWN WORKERS. CROPS COST BETWEEN 10 AND 15 RALLODS PER\n");
            print("SQUARE MILE TO PLANT.\n");
            print("YOUR GOAL IS TO COMPLETE YOUR " + n5 + " YEAR TERM OF OFFICE.\n");
            print("GOOD LUCK!\n");
        }
        # 打印换行
        print("\n");
        # 计算随机数并赋值给变量a
        a = Math.floor(60000 + (1000 * Math.random()) - (1000 * Math.random()));
        # 计算随机数并赋值给变量b
        b = Math.floor(500 + (10 * Math.random()) - (10 * Math.random()));
        # 变量c赋值为0
        c = 0;
        # 变量d赋值为2000
        d = 2000;
        # 变量x5赋值为0
        x5 = 0;
    }
    # 变量v3赋值为0
    v3 = 0;
    # 变量b5赋值为0
    b5 = 0;
    # 变量x赋值为false
    x = false;
    }
    # 如果随机数小于等于0.5，则执行以下代码块
    if (Math.random() <= 0.5) {
        print("YOU HAVE BEEN ASSASSINATED.\n");
    } else {
        # 否则执行以下代码块
        print("YOU HAVE BEEN THROWN OUT OF OFFICE AND ARE NOW\n");
        print("RESIDING IN PRISON.\n");
    }
    # 打印换行
    print("\n");
    print("\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```