#! /bin/bash

if [ -z "$1" ]
  then
    echo "No host supplied"
fi
if [ -z "$2" ]
  then
    echo "No nr_hosts supplied"
fi

host=$1
nr_hosts=$2

for i in $(seq 1 $(($nr_hosts)) ) ; do
    ssh -O stop -S "~/.ssh/master-socket/$(whoami)@${host}_${i}:22" $host
    ssh -O exit -S "~/.ssh/master-socket/$(whoami)@${host}_${i}:22" $host
done
