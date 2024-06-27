#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT
n_rnns=(10 15 20 25 30 64 128)
gpus=(0 1 2 3 4 5 6 7)

num_of_gpus=${#gpus[@]}
# seeds=(100 200 300 400 500 600 700 800 900 1000)
seeds=(1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)

task_num_list=(3 6 11 16 20)

for n_rnn in "${n_rnns[@]}"; do
    index=0
    for seed in "${seeds[@]}"; do
        for task_num in "${task_num_list[@]}"; do
            gpu=${gpus[$index]}
            log_dir="./runs/Fig2bcde_data/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
            echo "Launching task_num $task_num on GPU $gpu with seed $seed"
            # 确保日志目录存在
            mkdir -p $log_dir
            # 启动训练进程
            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.1 \
                --task_num $task_num \
                --gpu $gpu \
                --seed $seed \
                --log_dir $log_dir \
                --non_linearity relu \
                --max_trials 3000000 &
            let index+=1
            let index%=num_of_gpus
        done
    done
    wait  # 等待所有后台任务完成
    echo "All jobs for n_rnn=$n_rnn completed at $(date)"
done