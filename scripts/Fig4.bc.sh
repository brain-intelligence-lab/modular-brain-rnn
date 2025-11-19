#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

python main.py --gen_dataset_files --max_trials 3000000

# 捕获SIGINT信号
trap 'cleanup' SIGINT
n_rnn=84
gpus=(0 1 2 3 4 5 6 7)

num_of_gpus=${#gpus[@]}
seeds=($(seq 100 100 2000))  
eta=-3.0
conn_num=2000
index=0

for seed in "${seeds[@]}"; do
    gpu=${gpus[$index]}
    log_dir="./runs/Fig4bc/all-in-one/n_rnn_${n_rnn}_seed_${seed}"
    echo "Launching all-in-one mode on GPU $gpu with seed $seed"
    # 确保日志目录存在
    mkdir -p $log_dir
    # 启动训练进程
    python main.py \
        --n_rnn $n_rnn \
        --rec_scale_factor 0.1 \
        --gpu $gpu \
        --seed $seed \
        --log_dir $log_dir \
        --non_linearity relu \
        --init_mode randortho \
        --read_from_file \
        --eval_perf \
        --conn_mode fixed \
        --eta $eta \
        --conn_num $conn_num \
        --task_num 20 \
        --save_model \
        --max_trials 2560000 &
    let index+=1
    let index%=num_of_gpus


    gpu=${gpus[$index]}
    log_dir="./runs/Fig4bc/incremental/n_rnn_${n_rnn}_seed_${seed}"
    echo "Launching connection_incremental mode on GPU $gpu with seed $seed"
    # 确保日志目录存在
    mkdir -p $log_dir
    # 启动训练进程
    python main.py \
        --n_rnn $n_rnn \
        --rec_scale_factor 0.1 \
        --gpu $gpu \
        --seed $seed \
        --log_dir $log_dir \
        --non_linearity relu \
        --init_mode randortho \
        --read_from_file \
        --eval_perf \
        --conn_mode grow \
        --eta $eta \
        --conn_num $conn_num \
        --add_conn_per_stage $((conn_num/80)) \
        --max_steps_per_stage 500 \
        --task_num 20 \
        --save_model \
        --max_trials 2560000 &
    let index+=1
    let index%=num_of_gpus

done
echo "All jobs for conn_num=$conn_num started at $(date)"
wait  # 等待所有后台任务完成


