#!/usr/bin/env python
import subprocess


def build_environment():
    try:
        import tox
    except ImportError:
        tox = False

    if not tox:
        # Need to install tox
        subprocess.call(['pip', 'install', 'tox'])

    try:
        import numpy
    except ImportError:
        numpy = False

    if not numpy:
        # Need to install numpy
        subprocess.call(['pip', 'install', 'numpy'])


if __name__ == "__main__":
    build_environment()
