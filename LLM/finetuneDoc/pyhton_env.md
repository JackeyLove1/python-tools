sudo apt install python3.10
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 alpara-lora-venv-py310-cu117
source /home/guodong.li/virtual-venv/alpara-lora-venv-py310-cu117/bin/activate