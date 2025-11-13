
# 定义一个函数来处理信号
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # 杀死所有子进程
    exit
}

# 捕获SIGINT信号
trap 'cleanup' SIGINT


gpus=(0 1 2 3)
num_of_gpus=${#gpus[@]}


# steps=($(seq 500 100 1600))

# steps=($(seq 500 45000 500))

steps=(500 1000 5000 10000 20000 30000 40000 45000)

index=0

for step in "${steps[@]}"; do
        gpu=${gpus[$index]}
        echo "Launching step $step on GPU $gpu"

        python tmp2.py \
            --step $step \
            --gpu $gpu &  

        let index+=1
        let index%=num_of_gpus

done

wait  # 等待所有后台任务完成
echo "All jobs completed at $(date)"