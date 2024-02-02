# `basic-computer-games\57_Literature_Quiz\javascript\litquiz.js`

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
                       // 监听键盘事件，当按下回车键时，将输入的值传递给 resolve 函数
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的值
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
    print(tab(25) + "LITERATURE QUIZ\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量 r
    r = 0;
    // 输出提示信息
    print("TEST YOUR KNOWLEDGE OF CHILDREN'S LITERATURE.\n");
    print("\n");
    print("THIS IS A MULTIPLE-CHOICE QUIZ.\n");
    print("TYPE A 1, 2, 3, OR 4 AFTER THE QUESTION MARK.\n");
    print("\n");
    print("GOOD LUCK!\n");
}
    # 打印两个空行
    print("\n");
    print("\n");
    # 打印问题
    print("IN PINOCCHIO, WHAT WAS THE NAME OF THE CAT\n");
    # 打印选项
    print("1)TIGGER, 2)CICERO, 3)FIGARO, 4)GUIPETTO\n");
    # 获取用户输入并转换为整数
    a = parseInt(await input());
    # 判断用户输入是否正确，如果正确则打印提示并增加正确答案数量
    if (a == 3) {
        print("VERY GOOD!  HERE'S ANOTHER.\n");
        r++;
    } else {
        # 如果错误则打印提示
        print("SORRY...FIGARO WAS HIS NAME.\n");
    }
    # 打印两个空行
    print("\n");
    print("\n");
    # 重复上述过程，依次打印问题、选项、获取用户输入、判断答案是否正确，并根据结果打印相应提示
    # 最后根据正确答案数量打印不同的总结提示
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```