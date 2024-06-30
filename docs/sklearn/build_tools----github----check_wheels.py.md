# `D:\src\scipysrc\scikit-learn\build_tools\github\check_wheels.py`

```
# 导入系统库和路径操作库
import sys
from pathlib import Path

# 导入 YAML 库，用于加载 YAML 配置文件
import yaml

# 定义 GitHub Actions 中 wheels.yml 文件的路径
gh_wheel_path = Path.cwd() / ".github" / "workflows" / "wheels.yml"

# 打开并加载 wheels.yml 文件
with gh_wheel_path.open("r") as f:
    wheel_config = yaml.safe_load(f)

# 获取构建矩阵中的包含部分
build_matrix = wheel_config["jobs"]["build_wheels"]["strategy"]["matrix"]["include"]

# 计算构建的轮子数量
n_wheels = len(build_matrix)

# 添加一个额外的轮子数量用于 sdist
n_wheels += 1

# 定义 Cirrus CI 中 arm64 构建任务的配置文件路径
cirrus_path = Path.cwd() / "build_tools" / "cirrus" / "arm_wheel.yml"

# 打开并加载 arm_wheel.yml 文件
with cirrus_path.open("r") as f:
    cirrus_config = yaml.safe_load(f)

# 计算 Cirrus CI 中 arm64 构建任务的数量，并加到总轮子数量中
n_wheels += len(cirrus_config["linux_arm64_wheel_task"]["matrix"])

# 获取目录中 dist 文件夹下的所有文件列表
dist_files = list(Path("dist").glob("**/*"))

# 计算 dist 文件夹下文件的数量
n_dist_files = len(dist_files)

# 检查计算出的轮子数量与 dist 文件夹中文件数量是否一致
if n_dist_files != n_wheels:
    # 输出错误信息，指出预期的轮子数量与实际的文件数量不符
    print(
        f"Expected {n_wheels} wheels in dist/* but "
        f"got {n_dist_files} artifacts instead."
    )
    # 退出程序并返回状态码 1 表示错误
    sys.exit(1)

# 输出成功信息，显示 dist 文件夹中包含的轮子数量
print(f"dist/* has the expected {n_wheels} wheels:")
print("\n".join(file.name for file in dist_files))
```