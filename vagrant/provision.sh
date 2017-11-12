apt-get update
apt-get install -y build-essential checkinstall
apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

if [ "$(python2.7 -V 2>&1)" != "Python 2.7.13" ]
then
    cd /usr/src
    wget https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tgz

    tar xzf Python-2.7.13.tgz

    cd Python-2.7.13
    ./configure
    make altinstall
fi

if [ "$(python3.5 -V 2>&1)" != "Python 3.5.4" ]
then
    wget https://www.python.org/ftp/python/3.5.4/Python-3.5.4.tgz

    tar xzf Python-3.5.4.tgz

    cd Python-3.5.4
    ./configure
    make altinstall
fi

if [ "$(python3.6 -V 2>&1)" != "Python 3.6.3" ]
then
    wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz

    tar xzf Python-3.6.3.tgz

    cd Python-3.6.3
    ./configure
    make altinstall
fi

# Setup tox for running locally

pip3.6 install -U tox cython

rm -rf ~/mtfit
cp -r /mtfit ~/mtfit
cd ~/mtfit
pip3.6 install -r requirements.txt

tox -e py27, py35, py36, examples
