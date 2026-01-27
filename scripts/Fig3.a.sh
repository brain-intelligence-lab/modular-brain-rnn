#!/bin/bash

# Define a function to handle signals
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # Kill all child processes
    exit
}

# Capture SIGINT signal
trap 'cleanup' SIGINT

gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}
seeds=($(seq 100 100 2000))  


# go experiments:

n_rnns=(3 4 5 6 7 8 9 10)

task_list=("fdgo" 
            "fdanti"
            "delaygo"
            "fdgo fdanti"
            "fdgo delaygo"
            )


for seed in "${seeds[@]}"; do
    index=0
    for n_rnn in "${n_rnns[@]}"; do
        for task_name in "${task_list[@]}"; do
            gpu=${gpus[$index]}
            safe_task_name="${task_name// /_}"
            log_dir="./runs/Fig3a_go/n_rnn_${n_rnn}_task_${safe_task_name}_seed_${seed}"
            echo "Launching task_name $safe_task_name on GPU $gpu with seed $seed"
            # Ensure log directory exists
            mkdir -p $log_dir
            # Start training process
            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.1 \
                --task_list $task_name \
                --gpu $gpu \
                --seed $seed \
                --init_mode randortho \
                --log_dir $log_dir \
                --eval_perf \
                --non_linearity relu \
                --max_trials 3000000 &
            let index+=1
            let index%=num_of_gpus
        done
    done
    
    wait  # Wait for all background tasks to complete
    echo "All jobs for n_rnn=$n_rnn with seed=$seed completed at $(date)"
done



# contextdelaydm experiments:


task_list=("contextdelaydm1" "contextdelaydm2"
            "contextdelaydm1 contextdelaydm2")

n_rnns=(15 16 17 18 19 20 21 22 23 24)
for seed in "${seeds[@]}"; do
    index=0
    for n_rnn in "${n_rnns[@]}"; do
        for task_name in "${task_list[@]}"; do
            gpu=${gpus[$index]}
            safe_task_name="${task_name// /_}"
            log_dir="./runs/Fig3a_contextdelaydm_easy_task/n_rnn_${n_rnn}_task_${safe_task_name}_seed_${seed}"
            echo "Launching task_name $safe_task_name on GPU $gpu with seed $seed"
            # Ensure log directory exists
            mkdir -p $log_dir
            # Start training process
            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.1 \
                --task_list $task_name \
                --gpu $gpu \
                --seed $seed \
                --init_mode randortho \
                --eval_perf \
                --easy_task \
                --log_dir $log_dir \
                --non_linearity relu \
                --max_trials 3000000 &
            let index+=1
            let index%=num_of_gpus
        done
    done

    if (( seed % 100 == 0 )); then
        wait  # Wait for all background tasks to complete
        echo "All jobs for n_rnn=$n_rnn with seed<=$seed started at $(date)"
    fi
done



