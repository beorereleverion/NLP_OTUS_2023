#!/bin/bash

check_and_sleep() {
    current_hour=$(date +'%H')

    if (( current_hour >= 23 || current_hour < 8 )); then
        echo "Current time is between 23:00 and 8:00. Exiting."
        return
    else
        echo "Current time is outside 23:00 - 8:00. Sleeping for 5 minutes..."
        sleep 300  # 300 seconds = 5 minutes
        check_and_sleep  # Call itself
    fi
}

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep


export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep


export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep


export TRESHOLD="50"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="50"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-4"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-7"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="128"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="16"
python3.10 project-ruRoberta.py || exit

check_and_sleep

export TRESHOLD="100"
export LR_VALUE="2e-5"
export EPS_VALUE="1e-8"
export SENTENCE_LENGTH="256"
export BATCH_SIZE="32"
python3.10 project-ruRoberta.py || exit

check_and_sleep
