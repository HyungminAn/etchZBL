#!/bin/bash

# Usage: ./unmount.sh /path/to/archive.tar.gz
path_src=$1
path_dst=${path_src%.tar.gz}
mount_list="mount_list.txt"

# Unmount
fusermount -u "${path_dst}"

# mount_list에서 해당 경로 제거
current_host=$(hostname)
current_user=$(whoami)
key="${path_dst}, ${current_host}, ${current_user}"
sed -i "\|${key}|d" "${mount_list}"

