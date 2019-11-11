source .env
echo $sudo_password|sudo -S mongod --fork --logpath /var/log/mongodb.log  --bind_ip 192.168.1.73