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

n_rnns=(8 16 32 64)
task_num=20
gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}

mask_types=("prior_modular" "posteriori_modular" "random")
seeds=($(seq 100 100 2000))  

index=0
for seed in "${seeds[@]}"; do 
    for n_rnn in "${n_rnns[@]}"; do
        for mask_type in "${mask_types[@]}"; do
            gpu=${gpus[$index]}
            # Construct log_dir path
            log_dir="./runs/Fig4efg/lottery_ticket_hypo_${mask_type}/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
            load_model_path="./runs/Fig2b-h/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
            echo "Launching task_num $task_num on GPU $gpu with seed $seed"
            # Ensure log directory exists
            mkdir -p $log_dir
            # Start training process
            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.1 \
                --task_num $task_num \
                --gpu $gpu \
                --init_mode randortho \
                --seed $seed \
                --mod_lottery_hypo \
                --log_dir $log_dir \
                --eval_perf \
                --mask_type $mask_type \
                --save_model \
                --load_model_path $load_model_path \
                --read_from_file \
                --non_linearity relu \
                --max_trials 2560000 &  
            let index+=1
            let index%=num_of_gpus

        done
    done


    if (( seed % 400 == 0 )); then
        echo "All jobs for seed<=$seed started at $(date)"
        wait  # Wait for all background tasks to complete
    fi

done


