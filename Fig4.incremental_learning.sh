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
gpus=(0 1 2 3 4 5 6 7)

num_of_gpus=${#gpus[@]}
seeds=(100 200 300 400 500 600 700 800 900 1000)
# seeds=(1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)

reg_factor_list=(0.5 0.2 0.4 0.8)

conn_modes=("fix" "grow")

index=0
conn_num=2000
task_num=20

for n_rnn in "${n_rnns[@]}"; do

    for seed in "${seeds[@]}"; do

        gpu=${gpus[$index]}
        log_dir="./runs/Fig4_interleaved_learning/n_rnn_${n_rnn}_fixed_seed_${seed}"
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
            --conn_mode fixed \
            --conn_num $conn_num \
            --task_num $task_num \
            --max_trials 3000000 &
        let index+=1
        let index%=num_of_gpus


        gpu=${gpus[$index]}
        log_dir="./runs/Fig4_interleaved_learning/n_rnn_${n_rnn}_grow_seed_${seed}"
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
            --conn_mode grow \
            --conn_num $conn_num \
            --add_conn_per_stage $((conn_num/80)) \
            --max_steps_per_stage 500 \
            --task_num $task_num \
            --max_trials 3000000 &
        let index+=1
        let index%=num_of_gpus


        for reg_factor in "${reg_factor_list[@]}"; do

            gpu=${gpus[$index]}
            log_dir="./runs/Fig4_incremental_learning/n_rnn_${n_rnn}_regfactor_${reg_factor}_seed_${seed}"
            echo "Launching incremental learning on GPU $gpu with seed $seed"
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
                --continual_learning \
                --easy_task \
                --conn_mode grow \
                --conn_num $conn_num \
                --add_conn_per_stage $((conn_num/80)) \
                --max_steps_per_stage 500 \
                --reg_factor $reg_factor \
                --max_trials 640000 &
            let index+=1
            let index%=num_of_gpus
        done

    done
    wait  # 等待所有后台任务完成
    echo "All jobs for n_rnn=$n_rnn completed at $(date)"
done

