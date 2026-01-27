
# Define a function to handle signals
cleanup() {
    echo "Caught SIGINT signal. Cleaning up..."
    kill $(jobs -p)  # Kill all child processes
    exit
}

# Capture SIGINT signal
trap 'cleanup' SIGINT


gpus=(0 1 2 3 4 5 6 7)
num_of_gpus=${#gpus[@]}


# steps=($(seq 500 100 1600))

# steps=($(seq 500 45000 500))

steps=(500 1000 2000 5000 10000 20000 30000 40000)

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

wait  # Wait for all background tasks to complete
echo "All jobs completed at $(date)"