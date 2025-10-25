#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT

hc_s=(8 16 32 64)

class_nums=(2 10)
gpus=(0 1 2 3 4 5 6 7)

num_of_gpus=${#gpus[@]}

seeds=($(seq 100 100 1600))


for c1 in "${hc_s[@]}"; do
    
    index=0
    for class_num in "${class_nums[@]}"; do

        if [ "$class_num" -eq 2 ]; then   
            epochs=15
        else
            epochs=3
        fi

        for seed in "${seeds[@]}"; do
            gpu=${gpus[$index]}
            logdir="./runs/cnn_models_data/c1_${c1}_c2_${c1}_class_num_${class_num}_seed_${seed}"
            echo "Launching channels1 $c1 channels2 $c1 class_num $class_num on GPU $gpu with seed $seed"
            # 确保日志目录存在
            mkdir -p $logdir
            # 启动训练进程
            python cnn_models_exp.py \
                --channels_1 $c1 \
                --channels_2 $c1 \
                --gpu $gpu \
                --seed $seed \
                --logdir $logdir \
                --analyze_every_n_batches 40 \
                --epochs $epochs \
                --class_num $class_num \
                --weight_scale_factor 0.1 &
            let index+=1
            let index%=num_of_gpus
        done
    done
    wait  # 等待所有后台任务完成
    echo "All jobs for hidden_channel $h_c output_channel $o_c completed at $(date)"

done

hc_s=(8 16 32 64 128)
task_idxes=($(seq -1 1 11))
epochs=25

for hidden_channel in "${hc_s[@]}"; do
    for task_idx in "${task_idxes[@]}"; do
        index=0
        for seed in "${seeds[@]}"; do

            gpu=${gpus[$index]}
            logdir="./runs/gnn_models_data/hidden_channels_${hidden_channel}_task_idx_${task_idx}_seed_${seed}"
            echo "Launching hidden_channels $hidden_channel  task_idx $task_idx on GPU $gpu with seed $seed"
            # 确保日志目录存在
            mkdir -p $logdir
            # 启动训练进程
            python gnn_models_exp.py \
                --hidden_channels $hidden_channel \
                --gpu $gpu \
                --seed $seed \
                --logdir $logdir \
                --analyze_every_n_batches 40 \
                --epochs $epochs \
                --task_idx $task_idx \
                --weight_scale_factor 0.1 &
            let index+=1
            let index%=num_of_gpus
        done

    done

    wait  # 等待所有后台任务完成
    echo "All jobs for hidden_channel=$hidden_channel started at $(date)"

done