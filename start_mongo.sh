source .env
echo $sudo_password|sudo -S mongod --fork --logpath /var/log/mongodb.log