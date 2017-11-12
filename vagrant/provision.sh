apt-get update
apt-get install build-essential checkinstall
apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-de

cd /usr/src
wget https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tgz

tar xzf Python-2.7.13.tgz

cd Python-2.7.13
./configure
make altinstall

wget https://www.python.org/ftp/python/3.5.4/Python-3.5.4.tgz

tar xzf Python-3.5.4.tgz

cd Python-3.5.4
./configure
make altinstall
wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz

tar xzf Python-3.6.3.tgz

cd Python-3.6.3
./configure
make altinstall