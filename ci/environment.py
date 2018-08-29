#!/usr/bin/env python
import subprocess
import os


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
    if os.environ.get('ON_TRAVIS', None) == '1':
        subprocess.call(['pip', 'install', 'tox-travis'])


if __name__ == "__main__":
    build_environment()
