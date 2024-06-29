# `.\numpy\pavement.py`

```
r"""
This paver file is intended to help with the release process as much as
possible. It relies on virtualenv to generate 'bootstrap' environments as
independent from the user system as possible (e.g. to make sure the sphinx doc
is built against the built numpy, not an installed one).

Building changelog + notes
==========================

Assumes you have git and the binaries/tarballs in installers/::

    paver write_release
    paver write_note

This automatically put the checksum into README.rst, and writes the Changelog.

TODO
====
    - the script is messy, lots of global variables
    - make it more easily customizable (through command line args)
    - missing targets: install & test, sdist test, debian packaging
    - fix bdist_mpkg: we build the same source twice -> how to make sure we use
      the same underlying python for egg install in venv and for bdist_mpkg
"""
import os
import sys
import shutil
import hashlib
import textwrap

# The paver package needs to be installed to run tasks
import paver
from paver.easy import Bunch, options, task, sh

#-----------------------------------
# Things to be changed for a release
#-----------------------------------

# Path to the release notes
RELEASE_NOTES = 'doc/source/release/2.1.0-notes.rst'


#-------------------------------------------------------
# Hardcoded build/install dirs, virtualenv options, etc.
#-------------------------------------------------------

# Where to put the release installers
options(installers=Bunch(releasedir="release",
                         installersdir=os.path.join("release", "installers")),)


#-------------
# README stuff
#-------------

def _compute_hash(idirs, hashfunc):
    """Hash files using given hashfunc.

    Parameters
    ----------
    idirs : directory path
        Directory containing files to be hashed.
    hashfunc : hash function
        Function to be used to hash the files.

    """
    # List files in the specified directory
    released = paver.path.path(idirs).listdir()
    checksums = []
    # Iterate through each file path
    for fpath in sorted(released):
        # Open each file in binary mode
        with open(fpath, 'rb') as fin:
            # Compute the hash using the provided hash function
            fhash = hashfunc(fin.read())
            # Append checksum and file name to the list
            checksums.append(
                '%s  %s' % (fhash.hexdigest(), os.path.basename(fpath)))
    return checksums


def compute_md5(idirs):
    """Compute md5 hash of files in idirs.

    Parameters
    ----------
    idirs : directory path
        Directory containing files to be hashed.

    """
    # Delegate hash computation to _compute_hash using hashlib.md5
    return _compute_hash(idirs, hashlib.md5)


def compute_sha256(idirs):
    """Compute sha256 hash of files in idirs.

    Parameters
    ----------
    idirs : directory path
        Directory containing files to be hashed.

    """
    # Delegate hash computation to _compute_hash using hashlib.sha256
    # sha256 is preferred for better checksums
    return _compute_hash(idirs, hashlib.sha256)


def write_release_task(options, filename='README'):
    """Append hashes of release files to release notes.

    Parameters
    ----------
    options : paver.easy.Bunch
        Options object containing release configurations.
    filename : str, optional
        Name of the file to append hashes to (default is 'README').

    """
    This appends file hashes to the release notes and creates
    four README files of the result in various formats:

    - README.rst
    - README.rst.gpg
    - README.md
    - README.md.gpg

    The md file are created using `pandoc` so that the links are
    properly updated. The gpg files are kept separate, so that
    the unsigned files may be edited before signing if needed.

    Parameters
    ----------
    options :
        Set by ``task`` decorator.
        由 ``task`` 装饰器设置的选项。
    filename : str
        Filename of the modified notes. The file is written
        in the release directory.
        修改后的笔记文件名。文件将被写入发布目录。

    """
    idirs = options.installers.installersdir
    notes = paver.path.path(RELEASE_NOTES)
    rst_readme = paver.path.path(filename + '.rst')
    md_readme = paver.path.path(filename + '.md')

    # append hashes
    with open(rst_readme, 'w') as freadme:
        with open(notes) as fnotes:
            freadme.write(fnotes.read())

        # Write MD5 hashes to README.rst
        freadme.writelines(textwrap.dedent(
            """
            Checksums
            =========

            MD5
            ---
            ::

            """))
        freadme.writelines([f'    {c}\n' for c in compute_md5(idirs)])

        # Write SHA256 hashes to README.rst
        freadme.writelines(textwrap.dedent(
            """
            SHA256
            ------
            ::

            """))
        freadme.writelines([f'    {c}\n' for c in compute_sha256(idirs)])

    # generate md file using pandoc before signing
    sh(f"pandoc -s -o {md_readme} {rst_readme}")

    # Sign files
    if hasattr(options, 'gpg_key'):
        # Generate a clear-signed, armored ASCII file with default or specified GPG key
        cmd = f'gpg --clearsign --armor --default_key {options.gpg_key}'
    else:
        # Generate a clear-signed, armored ASCII file
        cmd = 'gpg --clearsign --armor'

    # Sign README.rst and README.md using GPG
    sh(cmd + f' --output {rst_readme}.gpg {rst_readme}')
    sh(cmd + f' --output {md_readme}.gpg {md_readme}')
# 使用 @task 装饰器声明一个任务函数，用于生成发布说明文件。
@task
# 定义函数 write_release，用于生成 README 文件。
def write_release(options):
    """Write the README files.

    Two README files are generated from the release notes, one in ``rst``
    markup for the general release, the other in ``md`` markup for the github
    release notes.

    Parameters
    ----------
    options :
        Set by ``task`` decorator.

    """
    # 获取发布目录路径
    rdir = options.installers.releasedir
    # 调用 write_release_task 函数，生成 README 文件，文件路径为发布目录下的 README
    write_release_task(options, os.path.join(rdir, 'README'))
```