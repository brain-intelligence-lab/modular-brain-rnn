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
seeds=($(seq 100 100 800))

task_list=("fdgo" "reactgo" "delaygo" "fdanti" "reactanti" "delayanti" 
              "dm1" "dm2" "contextdm1" "contextdm2" "multidm"
              "delaydm1" "delaydm2" "contextdelaydm1" "contextdelaydm2" "multidelaydm"
              "dmsgo" "dmsnogo" "dmcgo" "dmcnogo")



# rnn_types=("RNN")
# for n_rnn in "${n_rnns[@]}"; do
#     for task_name in "${task_list[@]}"; do
#         index=0
#         task_counter=0
#         for seed in "${seeds[@]}"; do
#             for rnn_type in "${rnn_types[@]}"; do
#                 gpu=${gpus[$index]}
#                 log_dir="./runs/Fig2a_data_${rnn_type}/n_rnn_${n_rnn}_task_${task_name}_seed_${seed}"
#                 echo "Launching task_name $task_name on GPU $gpu with seed $seed"
#                 # 确保日志目录存在
#                 mkdir -p $log_dir
#                 # 启动训练进程
#                 python main.py \
#                     --n_rnn $n_rnn \
#                     --rec_scale_factor 0.1 \
#                     --task_list $task_name \
#                     --rnn_type $rnn_type \
#                     --gpu $gpu \
#                     --seed $seed \
#                     --init_mode randortho \
#                     --log_dir $log_dir \
#                     --non_linearity relu \
#                     --max_trials 3000000 &
#                 let index+=1
#                 let index%=num_of_gpus
#                 let task_counter+=1
#             done
#         done
#         wait  # 等待所有后台任务完成
#         echo "All jobs for n_rnn=$n_rnn with seed<=$seed started at $(date)"
#     done
# done


rnn_types=("GRU" "LSTM")
n_rnns=(8 16 32)

for n_rnn in "${n_rnns[@]}"; do
    for task_name in "${task_list[@]}"; do
        index=0
        task_counter=0
        for seed in "${seeds[@]}"; do
            for rnn_type in "${rnn_types[@]}"; do
                gpu=${gpus[$index]}
                log_dir="./runs/Fig2a_data_${rnn_type}/n_rnn_${n_rnn}_task_${task_name}_seed_${seed}"
                echo "Launching task_name $task_name on GPU $gpu with seed $seed"
                # 确保日志目录存在
                mkdir -p $log_dir
                # 启动训练进程
                python main.py \
                    --n_rnn $n_rnn \
                    --rec_scale_factor 0.1 \
                    --task_list $task_name \
                    --rnn_type $rnn_type \
                    --gpu $gpu \
                    --seed $seed \
                    --init_mode randortho \
                    --log_dir $log_dir \
                    --non_linearity tanh \
                    --max_trials 2560000 &
                let index+=1
                let index%=num_of_gpus
                let task_counter+=1
            done
        done
        wait  # 等待所有后台任务完成
        echo "All jobs for n_rnn=$n_rnn with seed<=$seed started at $(date)"
    done
done

# for seed in "${seeds[@]}"; do
#     for n_rnn in "${n_rnns[@]}"; do
#         index=0
#         for task_name in "${task_list[@]}"; do
#             gpu=${gpus[$index]}
#             log_dir="./runs/Fig2a_data/chance_n_rnn_${n_rnn}_task_${task_name}_seed_${seed}"
#             echo "Launching task_name $task_name on GPU $gpu with seed $seed (at chance_level)"
#             # 确保日志目录存在
#             mkdir -p $log_dir
#             # 启动训练进程
#             python main.py \
#                 --n_rnn $n_rnn \
#                 --rec_scale_factor 0.1 \
#                 --task_list $task_name \
#                 --gpu $gpu \
#                 --seed $seed \
#                 --log_dir $log_dir \
#                 --non_linearity relu \
#                 --max_trials 3000000 \
#                 --get_chance_level &
#             let index+=1
#             let index%=num_of_gpus
#         done
#     done
#     wait  # 等待所有后台任务完成
#     echo "All jobs for n_rnn=$n_rnn completed at $(date)"
# done