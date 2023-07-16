import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
remote_host = "10.37.52.136"
ssh.connect(remote_host, username='fanhuanjie', password='')

stdin, stdout, stderr = ssh.exec_command('echo "Hello, World!"')

print(stdout.read().decode())
print(stderr.read().decode())

time.sleep(5) # wait for script to finish running

ssh.close()