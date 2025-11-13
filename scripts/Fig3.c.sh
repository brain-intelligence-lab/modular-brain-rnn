#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT

n_rnns=(8 64)
task_num=(20)
gpus=(0 1 2 3)
num_of_gpus=${#gpus[@]}
seeds=($(seq 65 1 256))  
act_fun='tanh'

index=0
for n_rnn in "${n_rnns[@]}"; do

    for seed in "${seeds[@]}"; do
        gpu=${gpus[$index]}
        log_dir="./runs/Fig3c_${act_fun}_${n_rnn}_new/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
        echo "Launching task_num $task_num on GPU $gpu with seed $seed"
        # 确保日志目录存在
        mkdir -p $log_dir
        # 启动训练进程
        python main.py \
            --n_rnn $n_rnn \
            --rec_scale_factor 0.01 \
            --task_num $task_num \
            --gpu $gpu \
            --display_step 20 \
            --init_mode one_init \
            --eval_perf \
            --seed $seed \
            --read_from_file \
            --log_dir $log_dir \
            --non_linearity $act_fun \
            --max_trials 2560000 &

        let index+=1
        let index%=num_of_gpus

        if (( seed % 16 == 0 )); then
            wait  # 等待所有后台任务完成
            echo "All jobs for n_rnn=$n_rnn with seed<=$seed started at $(date)"
        fi

    done
    
done
