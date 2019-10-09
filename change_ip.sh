#!/bin/bash

# First argument is old server
# Second argument is new server
vpn=IPVanish

sudo sed -i -e "s/$1/$2/g" /etc/NetworkManager/system-connections/$vpn
sudo service network-manager restart

while true; do
  if (service network-manager status | grep CONNECTED_GLOBAL); then
    nmcli con up $vpn
    break
  else
    echo 'Network Manager not initialzed yet. Retrying in 5...'
    sleep 5
  fi
done
