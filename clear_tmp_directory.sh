#!/bin/bash
source .env
cd /tmp/ || return
echo $sudo_password|sudo -S rm -r tmp* -f
