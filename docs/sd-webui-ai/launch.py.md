# `stable-diffusion-webui\launch.py`

```
# 从模块中导入 launch_utils 模块
from modules import launch_utils

# 从 launch_utils 模块中获取参数 args
args = launch_utils.args
# 从 launch_utils 模块中获取 python
python = launch_utils.python
# 从 launch_utils 模块中获取 git
git = launch_utils.git
# 从 launch_utils 模块中获取 index_url
index_url = launch_utils.index_url
# 从 launch_utils 模块中获取 dir_repos
dir_repos = launch_utils.dir_repos

# 从 launch_utils 模块中获取 commit_hash
commit_hash = launch_utils.commit_hash
# 从 launch_utils 模块中获取 git_tag
git_tag = launch_utils.git_tag

# 从 launch_utils 模块中获取 run
run = launch_utils.run
# 从 launch_utils 模块中获取 is_installed
is_installed = launch_utils.is_installed
# 从 launch_utils 模块中获取 repo_dir
repo_dir = launch_utils.repo_dir

# 从 launch_utils 模块中获取 run_pip
run_pip = launch_utils.run_pip
# 从 launch_utils 模块中获取 check_run_python
check_run_python = launch_utils.check_run_python
# 从 launch_utils 模块中获取 git_clone
git_clone = launch_utils.git_clone
# 从 launch_utils 模块中获取 git_pull_recursive
git_pull_recursive = launch_utils.git_pull_recursive
# 从 launch_utils 模块中获取 list_extensions
list_extensions = launch_utils.list_extensions
# 从 launch_utils 模块中获取 run_extension_installer
run_extension_installer = launch_utils.run_extension_installer
# 从 launch_utils 模块中获取 prepare_environment
prepare_environment = launch_utils.prepare_environment
# 从 launch_utils 模块中获取 configure_for_tests
configure_for_tests = launch_utils.configure_for_tests
# 从 launch_utils 模块中获取 start
start = launch_utils.start

# 定义主函数
def main():
    # 如果参数中包含 dump_sysinfo，则执行 dump_sysinfo() 函数，并将结果保存到 filename 中
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        # 打印保存的系统信息文件名，并退出程序
        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    # 记录初始启动时间
    launch_utils.startup_timer.record("initial startup")

    # 在 "prepare environment" 子类别下记录启动时间
    with launch_utils.startup_timer.subcategory("prepare environment"):
        # 如果参数中不包含 skip_prepare_environment，则执行 prepare_environment() 函数
        if not args.skip_prepare_environment:
            prepare_environment()

    # 如果参数中包含 test_server，则配置为测试环境
    if args.test_server:
        configure_for_tests()

    # 启动程序
    start()

# 如果当前脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```