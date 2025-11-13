#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT

n_rnns=(8 16 32 64)
task_num=20
gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}

mask_types=("prior_modular" "posteriori_modular" "random")
seeds=($(seq 100 100 1600))  

index=0
for n_rnn in "${n_rnns[@]}"; do
    for seed in "${seeds[@]}"; do
        for mask_type in "${mask_types[@]}"; do
            gpu=${gpus[$index]}
            # 构造 log_dir 路径
            log_dir="./runs/Fig4_lottery_ticket_hypo_${mask_type}/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
            load_model_path="./runs/Fig2bcde_data_one_init/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
            echo "Launching task_num $task_num on GPU $gpu with seed $seed"
            # 确保日志目录存在
            mkdir -p $log_dir
            # 启动训练进程
            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.01 \
                --task_num $task_num \
                --gpu $gpu \
                --init_mode one_init \
                --seed $seed \
                --mod_lottery_hypo \
                --log_dir $log_dir \
                --eval_perf \
                --mask_type $mask_type \
                --save_model \
                --load_model_path $load_model_path \
                --read_from_file \
                --non_linearity tanh \
                --max_trials 2560000 &  
            let index+=1
            let index%=num_of_gpus

        done
    done

    echo "All jobs submited!"
    wait  # 等待所有后台任务完成
    echo "All jobs for n_rnn=${n_rnn} for completed at $(date)"


done


