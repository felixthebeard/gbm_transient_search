#!/usr/bin/bash

TRIGGER_NAMES_FILE=$1
TRIGGER_DIR=$2
BALROG_TIMEOUT=$3

# Read the file containing the trigger names that is passed as argument
readarray -t trigger_names < ${TRIGGER_NAMES_FILE}

# run processes and store pids in array
n_triggers=${#trigger_names[@]}
n_cores=80
n_tasks=8

if [ "$n_triggers" -lt "$n_tasks" ]
then
    n_tasks="$n_triggers"
fi

python_path=python
script_path=/u/fkunzwei/scripts/bkg_pipe/run_balrog.py

# Calculate the number of cores that are used in the task
# this is an integer devision, that results in down rounding.
n_cores_per_task=$((n_cores / n_tasks))

count=0
while [ "$count" -lt "$n_triggers" ]; do

    # Start n_tasks and store the pids so we can track te process
    for i in $(seq 0 $(($n_tasks-1)) ) ; do

        # If we did not already reach the end of trigger_names then start a task
        if [ "$count" -lt "$n_triggers" ]
        then
	    trigger_name=${trigger_names[count]}
	    CMD="mpiexec -n $n_cores_per_task --timeout $BALROG_TIMEOUT $python_path $script_path $trigger_name $TRIGGER_DIR/$trigger_name/trigger_info.yml"
	    echo $CMD
            sleep 4 &
            pids[i]=$!
            #echo "mpi -n $n_cores_per_task ${trigger_names[count]} with pid $!"
        fi
        count=$((count+1))
    done

    # Wait for the n_tasks to complete before we start new ones
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
done

echo "Done"
