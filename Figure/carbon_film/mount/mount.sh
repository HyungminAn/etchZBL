#!/bin/bash

# Usage: ./mount.sh /path/to/archive.tar.gz
path_src=$1
path_dst=${path_src%.tar.gz}
path_index="${path_src}.index.sqlite"
mount_list="mount_list.txt"

current_host=$(hostname)
current_user=$(whoami)

# Mount using ratarmount
ratarmount --index-file "${path_index}" "${path_src}"

if [ $? -ne 0 ]; then
    echo "Failed to mount ${path_src}"
    exit 1
fi
echo "${path_dst}, ${current_host}, ${current_user}" >> "${mount_list}"
