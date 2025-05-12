#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT

n_rnn=84

task_num_list=(20)
gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}

wiring_rules=("random" "distance")

conn_modes=("fix" "grow")

conn_nums=(800)

seeds=($(seq 100 100 1000))

index=0

for conn_num in "${conn_nums[@]}"; do
    for task_num in "${task_num_list[@]}"; do
        for seed in "${seeds[@]}"; do
            for conn_mode in "${conn_modes[@]}"; do
                for wiring_rule in "${wiring_rules[@]}"; do
                    
                    gpu=${gpus[$index]}
                    # 构造 log_dir 路径
                    log_dir="./runs/Fig4b_data_2.0/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}_rule_${wiring_rule}_mode_${conn_mode}_conn_num_${conn_num}"
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
                        --add_conn_per_stage $((conn_num/80)) \
                        --max_steps_per_stage 500 \
                        --log_dir $log_dir \
                        --save_model \
                        --non_linearity relu \
                        --wiring_rule $wiring_rule \
                        --conn_mode $conn_mode \
                        --max_trials 3000000 &  
                
                    let index+=1
                    let index%=num_of_gpus
                done
            done
        done
    done

    echo "All jobs submited at $(date)"
    wait  # 等待所有后台任务完成
    echo "All jobs completed at $(date)"
done


echo "All jobs completed at $(date)"