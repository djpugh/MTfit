#!/usr/bin/env python
import subprocess
import os

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def test():
    subprocess.check_call(['python', '-m', 'tox'])
    #
    cwd = os.getcwd()
    os.chdir(REPO_PATH)
    subprocess.check_call(['docker', 'run', '-v', f'{cwd}:/src', 'python:27-slim', 'cd', '/src', '&', 'python', '-m', 'tox', '-e', 'test-py27,example-py27'])
    subprocess.check_call(['docker', 'run', '-v', f'{cwd}:/src', 'python:35-slim', 'cd', '/src', '&', 'python', '-m', 'tox', '-e', 'test-py35,example-py35'])
    subprocess.check_call(['docker', 'run', '-v', f'{cwd}:/src', 'python:36-slim', 'cd', '/src', '&', 'python', '-m', 'tox', '-e', 'test-py36,example-py36'])
    os.chdir(cwd)


if __name__ == "__main__":
    test()
