#!/bin/bash

# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT

hc_1s=(8 16 32 64 128 256)
hc_2s=(8 16 32 64 128 256)

# class_nums=(2 4 6 8 10)
class_nums=(2 10)
gpus=(0 1 2 3 4 5 6 7)

num_of_gpus=${#gpus[@]}
seeds=(100 200 300 400 500 600 700 800 900 1000)

for c1 in "${hc_1s[@]}"; do
    for c2 in "${hc_2s[@]}"; do
        index=0
        for class_num in "${class_nums[@]}"; do

            for seed in "${seeds[@]}"; do
                gpu=${gpus[$index]}
                log_dir="./runs/cnn_models_data/c1_${c1}_c2_${c2}_class_num_${class_num}_seed_${seed}"
                echo "Launching channels1 $c1 channels2 $c2 class_num $class_num on GPU $gpu with seed $seed"
                # 确保日志目录存在
                mkdir -p $log_dir
                # 启动训练进程
                python other_models_exp.py \
                    --channels_1 $c1 \
                    --channels_2 $c2 \
                    --gpu $gpu \
                    --seed $seed \
                    --log_dir $log_dir \
                    --analyze_every_n_batches 20 \
                    --epochs 1 \
                    --class_num $class_num \
                    --weight_scale_factor 0.1 &
                let index+=1
                let index%=num_of_gpus
            done
        done
        wait  # 等待所有后台任务完成
        echo "All jobs for hidden_channel $h_c output_channel $o_c completed at $(date)"

    done
done