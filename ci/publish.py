#!/usr/bin/env python
import os
import sys
import subprocess
import shutil
import glob

from pkg_resources import parse_version

try:
    import twine
except ImportError:
    subprocess.check_call(['pip', 'install', 'twine'])


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PYPI_URL = os.environ.get('TWINE_REPOSITORY_URL', None)


def publish(branch, pypi_url=PYPI_URL):
    wheel_files = glob.glob(os.path.join(ROOT_DIR, 'dist', '*.whl'))
    if wheel_files:
        wheel_file = sorted(wheel_files, key=lambda x: parse_version(x.split('-')[1]), reverse=True)[0]
    else:
        wheel_file = None
    if wheel_file is None:
        print('No wheels found')
        return
    else:
        print(f'Identified wheel: {wheel_file}')
        version = parse_version(os.path.split(wheel_file)[-1].split('-')[1])
        if version.local is None:
            # Only pushing tagged version to pypi
            args = ['python', '-m', 'twine', 'upload']
            if pypi_url is not None:
                args += ['--repository-url', pypi_url]
            args += ['--skip-existing', os.path.join(ROOT_DIR, 'dist', wheel_file)]
            subprocess.check_call(args)
            subprocess.check_call(['pipenv', 'run', 'tox', '-e', 'gh_pages'])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        publish_branch = sys.argv[1]
    else:
        publish_branch = 'master'
    if len(sys.argv) > 2:
        pypi_url = sys.argv[2]
    else:
        pypi_url = PYPI_URL
    # Let's check the current branch
    current_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().rstrip().rsplit('origin/')[-1]
    if current_branch == 'HEAD':
        current_branch = subprocess.check_output(['git', 'name-rev', '--name-only', 'HEAD']).decode().rstrip().rsplit('origin/')[-1]
    if current_branch == publish_branch:
        publish(publish_branch, pypi_url)
    else:
        print(f"Skipping as publish branch = {publish_branch} doesn't match the current branch = {current_branch}")