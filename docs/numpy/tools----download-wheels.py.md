# `.\numpy\tools\download-wheels.py`

```py
#!/usr/bin/env python3
"""
Script to download NumPy wheels from the Anaconda staging area.

Usage::

    $ ./tools/download-wheels.py <version> -w <optional-wheelhouse>

The default wheelhouse is ``release/installers``.

Dependencies
------------

- beautifulsoup4
- urllib3

Examples
--------

While in the repository root::

    $ python tools/download-wheels.py 1.19.0
    $ python tools/download-wheels.py 1.19.0 -w ~/wheelhouse

"""
import os
import re
import shutil
import argparse

import urllib3
from bs4 import BeautifulSoup

__version__ = "0.1"

# Edit these for other projects.
STAGING_URL = "https://anaconda.org/multibuild-wheels-staging/numpy"
PREFIX = "numpy"

# Name endings of the files to download.
WHL = r"-.*\.whl$"
ZIP = r"\.zip$"
GZIP = r"\.tar\.gz$"
SUFFIX = rf"({WHL}|{GZIP}|{ZIP})"


def get_wheel_names(version):
    """ Get wheel names from Anaconda HTML directory.

    This looks in the Anaconda multibuild-wheels-staging page and
    parses the HTML to get all the wheel names for a release version.

    Parameters
    ----------
    version : str
        The release version. For instance, "1.18.3".

    """
    # Create an HTTP connection pool manager with certificate validation
    http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED")
    # Regular expression pattern to match wheel file names for the given version
    tmpl = re.compile(rf"^.*{PREFIX}-{version}{SUFFIX}")
    # URL to the directory containing wheel files for the specified version
    index_url = f"{STAGING_URL}/files"
    # Perform a GET request to retrieve the HTML content of the directory
    index_html = http.request("GET", index_url)
    # Parse the HTML using BeautifulSoup for easier manipulation
    soup = BeautifulSoup(index_html.data, "html.parser")
    # Return all elements in the parsed HTML that match the wheel file name pattern
    return soup.find_all(string=tmpl)


def download_wheels(version, wheelhouse):
    """Download release wheels.

    The release wheels for the given NumPy version are downloaded
    into the given directory.

    Parameters
    ----------
    version : str
        The release version. For instance, "1.18.3".
    wheelhouse : str
        Directory in which to download the wheels.

    """
    # Create an HTTP connection pool manager with certificate validation
    http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED")
    # Get the list of wheel names for the specified NumPy version
    wheel_names = get_wheel_names(version)

    # Iterate through each wheel name and download it
    for i, wheel_name in enumerate(wheel_names):
        # Construct the URL to download the wheel file
        wheel_url = f"{STAGING_URL}/{version}/download/{wheel_name}"
        # Construct the local path where the wheel file will be saved
        wheel_path = os.path.join(wheelhouse, wheel_name)
        # Open a file in binary write mode to save the downloaded content
        with open(wheel_path, "wb") as f:
            # Perform a GET request to download the wheel file
            with http.request("GET", wheel_url, preload_content=False) as r:
                # Print the progress of each download
                print(f"{i + 1:<4}{wheel_name}")
                # Copy the downloaded content to the local file
                shutil.copyfileobj(r, f)
    # Print the total number of files downloaded
    print(f"\nTotal files downloaded: {len(wheel_names)}")


if __name__ == "__main__":
    # Initialize argument parser for command line arguments
    parser = argparse.ArgumentParser()
    # Positional argument: NumPy version to download
    parser.add_argument(
        "version",
        help="NumPy version to download.")
    # Optional argument: directory where downloaded wheels will be stored
    parser.add_argument(
        "-w", "--wheelhouse",
        default=os.path.join(os.getcwd(), "release", "installers"),
        help="Directory in which to store downloaded wheels\n"
             "[defaults to <cwd>/release/installers]")

    # Parse command line arguments
    args = parser.parse_args()

    # Expand user directory in case of '~' in the path
    wheelhouse = os.path.expanduser(args.wheelhouse)
    # 如果指定的 wheelhouse 目录不存在，则抛出运行时错误
    if not os.path.isdir(wheelhouse):
        # 使用 f-string 格式化错误信息，提示指定的 wheelhouse 目录不存在
        raise RuntimeError(
            f"{wheelhouse} wheelhouse directory is not present."
            " Perhaps you need to use the '-w' flag to specify one.")
    
    # 调用函数下载指定版本的 wheels 到指定的 wheelhouse 目录
    download_wheels(args.version, wheelhouse)
```