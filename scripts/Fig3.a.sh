#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT


n_rnns=(26 28 30 44 46 48 50 52 54)
gpus=(0 1 2 3 4 5 6 7)

num_of_gpus=${#gpus[@]}
seeds=(100 200 300 400 500 600 700 800)

# task_list=("contextdm1" "contextdm2"
#             "contextdm1 contextdm2"
#             )

# for seed in "${seeds[@]}"; do
#     index=0
#     for n_rnn in "${n_rnns[@]}"; do
#         for task_name in "${task_list[@]}"; do
#             gpu=${gpus[$index]}
#             safe_task_name="${task_name// /_}"
#             log_dir="./runs/Fig3a_contextdm/n_rnn_${n_rnn}_task_${safe_task_name}_seed_${seed}"
#             echo "Launching task_name $safe_task_name on GPU $gpu with seed $seed"
#             # 确保日志目录存在
#             mkdir -p $log_dir
#             # 启动训练进程
#             python main.py \
#                 --n_rnn $n_rnn \
#                 --rec_scale_factor 0.01 \
#                 --task_list $task_name \
#                 --gpu $gpu \
#                 --seed $seed \
#                 --init_mode one_init \
#                 --eval_perf \
#                 --log_dir $log_dir \
#                 --non_linearity tanh \
#                 --max_trials 3000000 &
#             let index+=1
#             let index%=num_of_gpus
#         done
#     done
#     wait  # 等待所有后台任务完成
#     # echo "All jobs for n_rnn=$n_rnn with seed=$seed completed at $(date)"
# done



n_rnns=(4 6 10)
gpus=(0 1 2 3)

num_of_gpus=${#gpus[@]}
seeds=(100 200 300 400 500 600 700 800)

task_list=("fdgo" 
            "fdanti"
            "delaygo"
            "fdgo fdanti"
            "fdgo delaygo"
            )


for seed in "${seeds[@]}"; do
    index=0
    for n_rnn in "${n_rnns[@]}"; do
        for task_name in "${task_list[@]}"; do
            gpu=${gpus[$index]}
            safe_task_name="${task_name// /_}"
            log_dir="./runs/Fig3a_go/n_rnn_${n_rnn}_task_${safe_task_name}_seed_${seed}"
            echo "Launching task_name $safe_task_name on GPU $gpu with seed $seed"
            # 确保日志目录存在
            mkdir -p $log_dir
            # 启动训练进程
            python main.py \
                --n_rnn $n_rnn \
                --rec_scale_factor 0.1 \
                --task_list $task_name \
                --gpu $gpu \
                --seed $seed \
                --init_mode randortho \
                --log_dir $log_dir \
                --eval_perf \
                --non_linearity relu \
                --max_trials 3000000 &
            let index+=1
            let index%=num_of_gpus
        done
    done
    
    wait  # 等待所有后台任务完成
    echo "All jobs for n_rnn=$n_rnn with seed=$seed completed at $(date)"
done

