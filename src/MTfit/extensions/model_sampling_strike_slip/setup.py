"""
Setup script for strike_slip
****************************
Call from command line as: sudo python setup.py install if on unix
or python setup.py install

"""
try:
    from setuptools import setup
except Exception:
    from distutils.core import setup


def setup_package():
    """
    setup_package()

    Function to call distutils setup to install the package to the default location for third party modules.

    Checks to see if required modules are installed and if not tries to install them.
    """
    kwargs = dict(name='model_sampling_strike_slip', version='1', author='David J Pugh', author_email='david.j.pugh@cantab.net',
                  packages=['model_sampling_strike_slip'],
                  package_dir={'model_sampling_strike_slip': '.'},
                  requires=['numpy', 'MTfit'],
                  install_requires=['numpy>=1.7.0'],
                  provides=['model_sampling_strike_slip'],
                  scripts=[],
                  description='model_sampling_strike_slip: Strike-slip random sampling for MTfit',
                  package_data={'strike_slip': []},)
    kwargs['entry_points'] = {'MTfit.sample_distribution': ['strike_slip= model_sampling_strike_slip:random_strike_slip']}
    setup(**kwargs)


if __name__ == "__main__":
    setup_package()
