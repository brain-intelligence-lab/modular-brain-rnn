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
gpus=(0 1 2 3 4 5 6 7)

num_of_gpus=${#gpus[@]}
seeds=($(seq 100 100 2000))
task_num_list=(3 6 11 16 20)

index=0
for seed in "${seeds[@]}"; do
    for task_num in "${task_num_list[@]}"; do
        for n_rnn in "${n_rnns[@]}"; do
            gpu=${gpus[$index]}
            log_dir="./runs/Fig2_b-h/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
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
                --init_mode randortho \
                --non_linearity relu \
                --save_model \
                --eval_perf \
                --max_trials 3000000 &
            let index+=1
            let index%=num_of_gpus
        done
    done

    if (( seed % 200 == 0 )); then
        echo "All jobs for n_rnn=$n_rnn with seed<=$seed started at $(date)"
        wait  # 等待所有后台任务完成
        index=0
    fi

done



# chance level experiment

# for n_rnn in "${n_rnns[@]}"; do
#     for seed in "${seeds[@]}"; do
#         index=0
#         for task_num in "${task_num_list[@]}"; do
#             gpu=${gpus[$index]}
#             log_dir="./runs/Fig2_b-h/chance_n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
#             echo "Launching task_num $task_num on GPU $gpu with seed $seed (at chance level)"
#             # 确保日志目录存在
#             mkdir -p $log_dir
#             # 启动训练进程
#             python main.py \
#                 --n_rnn $n_rnn \
#                 --rec_scale_factor 0.1 \
#                 --task_num $task_num \
#                 --gpu $gpu \
#                 --seed $seed \
#                 --init_mode randortho \
#                 --log_dir $log_dir \
#                 --non_linearity relu \
#                 --save_model \
#                 --max_trials 3000000 \
#                 --get_chance_level &
#             let index+=1
#             let index%=num_of_gpus
#         done

#         echo "All jobs for n_rnn=$n_rnn seed=$seed started at $(date)"
#         wait  # 等待所有后台任务完成
#     done
# done



# supplementary experiment: alternative RNN variants

# seeds=($(seq 100 100 1000))
# rnn_types=("GRU" "LSTM")
# n_rnns=(8 16 32 64)

# for seed in "${seeds[@]}"; do
#     index=0
#     for task_num in "${task_num_list[@]}"; do
#         for n_rnn in "${n_rnns[@]}"; do
#             for rnn_type in "${rnn_types[@]}"; do
#                 gpu=${gpus[$index]}
#                 log_dir="./runs/Fig2_b-h_${rnn_type}/n_rnn_${n_rnn}_task_${task_num}_seed_${seed}"
#                 echo "Launching task_num $task_num on GPU $gpu with seed $seed"
#                 # 确保日志目录存在
#                 mkdir -p $log_dir
#                 # 启动训练进程
#                 python main.py \
#                     --n_rnn $n_rnn \
#                     --rec_scale_factor 0.1 \
#                     --task_num $task_num \
#                     --gpu $gpu \
#                     --seed $seed \
#                     --rnn_type $rnn_type \
#                     --log_dir $log_dir \
#                     --init_mode randortho \
#                     --non_linearity tanh \
#                     --save_model \
#                     --eval_perf \
#                     --max_trials 2560000 &
#                 let index+=1
#                 let index%=num_of_gpus
#             done
#         done
#     done

#     echo "All jobs for n_rnn=$n_rnn with seed=$seed started at $(date)"
#     wait  # 等待所有后台任务完成
    
# done