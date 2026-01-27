#!/bin/bash

# Define a function to handle signals
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # Kill all child processes
    exit
}

# Capture SIGINT signal
trap 'cleanup' SIGINT

python main.py --gen_dataset_files --max_trials 3000000


n_rnns=(84)
task_num=20
gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}
seeds=($(seq 1 1 48))

index=0
for n_rnn in "${n_rnns[@]}"; do
    for seed in "${seeds[@]}"; do
        gpu=${gpus[$index]}

        log_dir="./runs/Fig5_data/${n_rnn}/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
        echo "Launching task_num $task_num on GPU $gpu with seed $seed"
        # Ensure log directory exists
        mkdir -p $log_dir
        # Start training process
        python main.py \
            --n_rnn $n_rnn \
            --rec_scale_factor 0.1 \
            --task_num $task_num \
            --gpu $gpu \
            --seed $seed \
            --log_dir $log_dir \
            --init_mode randortho \
            --non_linearity relu \
            --read_from_file \
            --eval_perf \
            --save_model \
            --reg_factor 0.000001 \
            --reg_term \
            --max_trials 3000000 &

        let index+=1
        let index%=num_of_gpus
    

        if (( seed % 24 == 0 )); then
            wait  # Wait for all background tasks to complete
            echo "All jobs for n_rnn=$n_rnn with seed<=$seed completed at $(date)"
        fi

    done
done
