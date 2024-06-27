#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT

n_rnn=64
task_num_list=(20)
gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}

conn_nums=(320 160 80 )

seeds=($(seq 1 100))  

index=0
for conn_num in "${conn_nums[@]}"; do
    for task_num in "${task_num_list[@]}"; do
        for seed in "${seeds[@]}"; do
            gpu=${gpus[$index]}
            # 构造 log_dir 路径
            log_dir="./runs/Fig4_topology_task/${n_rnn}_${conn_num}_${task_num}/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}_rule_random_mode_fix_conn_num_${conn_num}"
            echo "Launching task_num $task_num conn_num $conn_num on GPU $gpu with seed $seed"
            # 确保日志目录存在
            mkdir -p $log_dir
            # 启动训练进程

            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.1 \
                --task_num $task_num \
                --gpu $gpu \
                --seed $seed \
                --conn_num $conn_num \
                --add_conn_per_stage 0 \
                --log_dir $log_dir \
                --save_model \
                --non_linearity relu \
                --wiring_rule random \
                --conn_mode fix \
                --max_trials 2000000 &  
        
            let index+=1
            let index%=num_of_gpus

        done

        echo "All jobs with conn_num = $conn_num submited at $(date)"
        wait  # 等待所有后台任务完成
        echo "All jobs with conn_num = $conn_num completed at $(date)"

    done
done




n_rnn=32
task_num_list=(3 6 11 20)
gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}

conn_nums=(160 80 40)

seeds=($(seq 1 100))  

index=0

for conn_num in "${conn_nums[@]}"; do
    for task_num in "${task_num_list[@]}"; do
        for seed in "${seeds[@]}"; do
            gpu=${gpus[$index]}
            # 构造 log_dir 路径
            log_dir="./runs/Fig4_topology_task/${n_rnn}_${conn_num}_${task_num}/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}_rule_random_mode_fix_conn_num_${conn_num}"
            echo "Launching task_num $task_num conn_num $conn_num on GPU $gpu with seed $seed"
            # 确保日志目录存在
            mkdir -p $log_dir
            # 启动训练进程

            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.1 \
                --task_num $task_num \
                --gpu $gpu \
                --seed $seed \
                --conn_num $conn_num \
                --add_conn_per_stage 0 \
                --log_dir $log_dir \
                --save_model \
                --non_linearity relu \
                --wiring_rule random \
                --conn_mode fix \
                --max_trials 2000000 &  
        
            let index+=1
            let index%=num_of_gpus

        done

        echo "All jobs with conn_num = $conn_num submited at $(date)"
        wait  # 等待所有后台任务完成
        echo "All jobs with conn_num = $conn_num completed at $(date)"

    done
done


echo "All jobs completed at $(date)"


