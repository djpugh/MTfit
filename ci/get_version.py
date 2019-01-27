import subprocess
import sys

import pkg_resources

version = pkg_resources.parse_version(subprocess.check_output([sys.executable, 'setup.py', '--version']).decode())
if version.local:
    print('')
    sys.exit(1)
else:
    print(version)
    sys.exit(0)
