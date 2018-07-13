#!/usr/bin/env python
import subprocess
import os

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def build():
    # Build sdist into
    cwd = os.getcwd()

    if len(REPO_PATH):
        os.chdir(REPO_PATH)
    subprocess.check_call(['tox', '-e', 'build-py27,build-py35,build-py36'])
    os.chdir(cwd)


if __name__ == "__main__":
    build()
