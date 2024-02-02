# `arknights-mower\arknights_mower\utils\update.py`

```py
# 导入日志、操作系统、压缩文件、网络请求模块
import logging
import os
import zipfile
import requests
from .. import __version__


# 编写bat脚本，删除旧程序，运行新程序
def __write_restart_cmd(new_name, old_name):
    # 打开升级脚本文件，写入内容
    b = open("upgrade.bat", 'w')
    TempList = "@echo off\n"  # 关闭命令行窗口显示
    TempList += "if not exist " + new_name + " exit \n"  # 如果新版本程序不存在，则退出更新
    TempList += "echo 正在更新至最新版本...\n"  # 显示更新提示
    TempList += "timeout /t 5 /nobreak\n"  # 等待5秒
    TempList += "if exist " + old_name + ' del "' + old_name.replace("/", "\\\\") + '"\n'  # 删除旧程序
    TempList += 'copy  "' + new_name.replace("/", "\\\\") + '" "' + old_name.replace("/", "\\\\") + '"\n'  # 复制新版本程序
    TempList += "echo 更新完成，正在启动...\n"  # 显示更新完成提示
    TempList += "timeout /t 3 /nobreak\n"  # 等待3秒
    TempList += 'start  ' + old_name + ' \n'  # 启动旧程序
    TempList += "exit"  # 退出命令行
    b.write(TempList)  # 写入bat脚本内容
    b.close()  # 关闭文件
    # subprocess.Popen("upgrade.bat") #不显示cmd窗口
    os.system('start upgrade.bat')  # 显示cmd窗口
    os._exit(0)  # 退出程序


# 与github上最新版比较
def compere_version():
    """
        与github上最新版比较
        :return res: str | None, 若需要更新 返回版本号, 否则返回None
    """
    newest_version = __get_newest_version()  # 获取最新版本号

    v1 = [str(x) for x in str(__version__).split('.')]  # 将当前版本号拆分成列表
    v2 = [str(x) for x in str(newest_version).split('.')]  # 将最新版本号拆分成列表

    # 如果2个版本号位数不一致，后面使用0补齐，使2个list长度一致，便于后面做对比
    if len(v1) > len(v2):
        v2 += [str(0) for x in range(len(v1) - len(v2))]
    elif len(v1) < len(v2):
        v1 += [str(0) for x in range(len(v2) - len(v1))]
    list_sort = sorted([v1, v2])  # 对版本号列表进行排序
    if list_sort[0] == list_sort[1]:  # 如果两个版本号相同
        return None  # 返回None，无需更新
    elif list_sort[0] == v1:  # 如果当前版本号较小
        return newest_version  # 返回最新版本号，需要更新
    else:  # 如果当前版本号较大
        return None  # 返回None，无需更新


# 更新版本
def update_version():
    if os.path.isfile("upgrade.bat"):  # 如果存在升级脚本文件
        os.remove("upgrade.bat")  # 删除升级脚本文件
    __write_restart_cmd("tmp/mower.exe", "./mower.exe")  # 调用编写升级脚本的函数


# 获取最新版本号
def __get_newest_version():
    response = requests.get("https://api.github.com/repos/ArkMowers/arknights-mower/releases/latest")  # 发送GET请求获取最新版本信息
    return response.json()["tag_name"]  # 返回最新版本号


def download_version(version):
    # 如果"./tmp"目录不存在，则创建该目录
    if not os.path.isdir("./tmp"):
        os.makedirs("./tmp")
    # 从指定 URL 下载文件，stream=True 表示以流的方式下载
    r = requests.get(f"https://github.com/ArkMowers/arknights-mower/releases/download/{version}/mower.zip",stream=True)
    # 获取下载文件的总大小
    total = int(r.headers.get('content-length', 0))
    index = 0
    # 以二进制写入模式打开文件，将下载的内容写入文件
    with open('./tmp/mower.zip', 'wb') as f:
        # 以指定大小迭代下载的内容，写入文件
        for chunk in r.iter_content(chunk_size=10485760):
            if chunk:
                f.write(chunk)
                index += len(chunk)
                # 打印更新进度
                print(f"更新进度：{'%.2f%%' % (index*100 / total)}({index}/{total})")
    # 打开下载的 ZIP 文件
    zip_file = zipfile.ZipFile("./tmp/mower.zip")
    # 获取 ZIP 文件中的文件列表
    zip_list = zip_file.namelist()
    # 遍历 ZIP 文件中的文件列表，将文件解压到"./tmp/"目录下
    for f in zip_list:
        zip_file.extract(f, './tmp/')
    # 关闭 ZIP 文件
    zip_file.close()
    # 删除下载的 ZIP 文件
    os.remove("./tmp/mower.zip")
# 定义主函数
def main():
    # 如果存在名为"upgrade.bat"的文件，则删除该文件
    if os.path.isfile("upgrade.bat"):
        os.remove("upgrade.bat")
    # 调用__write_restart_cmd函数，传入参数"newVersion.exe"和"Version.exe"
    __write_restart_cmd("newVersion.exe", "Version.exe")

# 如果当前脚本作为主程序执行，则调用compere_version函数
if __name__ == '__main__':
    compere_version()
```