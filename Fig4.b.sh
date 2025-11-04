#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT
n_rnns=(84)
gpus=(0 1 2 3)

num_of_gpus=${#gpus[@]}
# seeds=(100 200 300 400 500 600 700 800)
seeds=(900 1000 1100 1200 1300 1400 1500 1600)
conn_modes=("fix" "grow")

index=0
conn_num=2000
task_num=20

for n_rnn in "${n_rnns[@]}"; do

    for seed in "${seeds[@]}"; do

        gpu=${gpus[$index]}
        log_dir="./runs/Fig4_interleaved_learning_fixed_reset/n_rnn_${n_rnn}_seed_${seed}"
        echo "Launching interleaved learning on GPU $gpu with seed $seed"
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
            --save_model \
            --read_from_file \
            --eval_perf \
            --conn_mode fixed \
            --conn_num $conn_num \
            --task_num $task_num \
            --max_trials 2560000 &
        let index+=1
        let index%=num_of_gpus


        gpu=${gpus[$index]}
        log_dir="./runs/Fig4_incremental_learning_alltask_reset/n_rnn_${n_rnn}_seed_${seed}"
        echo "Launching incremental learning alltask on GPU $gpu with seed $seed"
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
            --save_model \
            --read_from_file \
            --eval_perf \
            --conn_mode grow \
            --conn_num $conn_num \
            --add_conn_per_stage $((conn_num/80)) \
            --max_steps_per_stage 500 \
            --task_num $task_num \
            --max_trials 2560000 &
        let index+=1
        let index%=num_of_gpus

    done
    wait  # 等待所有后台任务完成
    echo "All jobs for n_rnn=$n_rnn completed at $(date)"
done

