#!/bin/bash

# First arg is new server
vpn=IPVanish

sed -i -e "s/gateway=.*/gateway=$1/g" /etc/NetworkManager/system-connections/$vpn
service network-manager restart

sleep 2

while true; do
  if (service network-manager status | grep 'CONNECTED_GLOBAL'); then
    nmcli con up $vpn
    if (service network-manager status | grep 'VPN connection: failed to connect:'); then
      echo "VPN address invalid ($1)."
      exit 1
    elif (service network-manager status | grep 'VPN connection: (IP Config Get) complete'); then
      break
    else
      echo "Neither string found for $1. Retrying in 5..."
      sleep 5
    fi
  else
    echo "Network Manager not initialized yet for $1. Retrying in 5..."
    service network-manager restart
    sleep 10
  fi
done