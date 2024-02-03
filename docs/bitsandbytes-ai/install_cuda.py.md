# `bitsandbytes\install_cuda.py`

```py
# 导入必要的库
import os
import subprocess
import sys
from urllib.request import urlretrieve

# CUDA 版本与对应的下载链接字典
cuda_versions = {
    "92": "https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux",
    "100": "https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux",
    "101": "https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run",
    "102": "https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run",
    "110": "https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run",
    "111": "https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run",
    "112": "https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run",
    "113": "https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run",
    "114": "https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run",
    "115": "https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run",
    "116": "https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run",
    "117": "https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run",
    "118": "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run",
    "120": "https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run",
    "121": "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run",
    # 键为"122"，值为CUDA 12.2.0 Linux 版本的下载链接
    "122": "https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run",
    # 键为"123"，值为CUDA 12.3.1 Linux 版本的下载链接
    "123": "https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda_12.3.1_545.23.08_linux.run",
}

# 安装 CUDA 版本
def install_cuda(version, base_path, download_path):
    # 格式化 CUDA 版本号，将最后一个字符前面的部分和最后一个字符拼接成新的版本号
    formatted_version = f"{version[:-1]}.{version[-1]}"
    # 根据格式化后的版本号创建 CUDA 文件夹名
    folder = f"cuda-{formatted_version}"
    # 拼接安装路径
    install_path = os.path.join(base_path, folder)

    # 如果安装路径已存在，则删除已存在的 CUDA 版本
    if os.path.exists(install_path):
        print(f"Removing existing CUDA version {version} at {install_path}...")
        subprocess.run(["rm", "-rf", install_path], check=True)

    # 获取 CUDA 版本对应的下载链接和文件名
    url = cuda_versions[version]
    filename = url.split('/')[-1]
    filepath = os.path.join(download_path, filename)

    # 如果下载路径不存在，则下载 CUDA 安装文件
    if not os.path.exists(filepath):
        print(f"Downloading CUDA version {version} from {url}...")
        urlretrieve(url, filepath)
    else:
        print(f"Installer for CUDA version {version} already downloaded.")

    # 将安装文件设置为可执行
    subprocess.run(["chmod", "+x", filepath], check=True)

    # 安装 CUDA
    print(f"Installing CUDA version {version}...")
    install_command = [
        "bash", filepath,
        "--no-drm", "--no-man-page", "--override",
        "--toolkitpath=" + install_path, "--toolkit", "--silent"
    ]

    print(f"Running command: {' '.join(install_command)}")

    try:
        subprocess.run(install_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Installation failed for CUDA version {version}: {e}")
        return
    finally:
        # 删除安装文件
        os.remove(filepath)

    print(f"CUDA version {version} installed at {install_path}")

# 主函数
def main():
    user_base_path = os.path.expanduser("~/cuda")
    system_base_path = "/usr/local/cuda"
    base_path = user_base_path  # 默认安装到用户目录
    download_path = "/tmp"  # 默认下载路径

    # 检查命令行参数是否足够
    if len(sys.argv) < 2:
        print("Usage: python install_cuda.py <version/all> [user/system] [download_path]")
        sys.exit(1)

    version = sys.argv[1]
    if len(sys.argv) > 2:
        base_path = system_base_path if sys.argv[2] == "system" else user_base_path
    # 检查命令行参数是否大于3，如果是则将第四个参数作为下载路径
    if len(sys.argv) > 3:
        download_path = sys.argv[3]

    # 如果基本路径不存在，则创建
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # 如果下载路径不存在，则创建
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # 安装 CUDA 版本
    # 如果版本是 "all"，则安装所有 CUDA 版本
    if version == "all":
        for ver in cuda_versions.keys():
            install_cuda(ver, base_path, download_path)
    # 如果版本在 CUDA 版本列表中，则安装指定版本
    elif version in cuda_versions:
        install_cuda(version, base_path, download_path)
    # 如果版本不在 CUDA 版本列表中，则打印错误信息并退出程序
    else:
        print(f"Invalid CUDA version: {version}. Available versions are: {', '.join(cuda_versions.keys())}")
        sys.exit(1)
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```