#!/usr/bin/env python
import subprocess


def lint():
    subprocess.check_call(['python', '-m', 'tox', '-e', 'lint-flake8,lint-check'])


if __name__ == "__main__":
    lint()
