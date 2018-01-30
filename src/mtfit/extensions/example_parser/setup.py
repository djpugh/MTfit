"""Setup script for example_parser
*********************************
Call from command line as: sudo python setup.py install if on unix
or python setup.py install


"""
try:
    from setuptools import setup
except Exception:
    from distutils.core import setup


def setup_package(install=False, test=False, build=False, develop=False, clean=False):
    """setup_package()

    Function to call distutils setup to install the package to the default location for third party modules.
    """
    kwargs = dict(name='example_parser', version='1', author='David J Pugh', author_email='david.j.pugh@cantab.net',
                  packages=['example_parser'],
                  package_dir={'example_parser': '.'},
                  requires=['MTfit'],
                  provides=['example_parser'],
                  description='example_parser: Example parser as an extension',
                  package_data={'example_parser': []},)
    kwargs['version'] = '1.0.0'
    kwargs['entry_points'] = {'MTfit.parsers': ['example_parser= example_parser:simple_parser']}
    setup(**kwargs)


if __name__ == "__main__":
    setup_package()
