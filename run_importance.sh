#/bin/bash

export LD_LIBRARY_PATH=/cm/shared/apps/cuda10.0/toolkit/current/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

: ${total_iterations:=1}
: ${start_iter:=0}
: ${dataset:="mnist_data"} # Choices: ["mnist_data", "mnist_per_class_data", "cifar_data", "cifar_per_class_data"]
: ${net_arch:="lenet"} # Choices: ["lenet", "subflow_alexnet"]
: ${num_classes:=10} # Choices: [10, 2]
: ${main_prog:="subflow_v1.py"}

if [ ${net_arch} = "lenet" ]; then
    net_arch_conf="28*28*1,5*5*1*6,5*5*6*16,400,84,${num_classes}" # LeNet-5 for MNIST
else
    net_arch_conf="32*32*3,3*3*3*64,3*3*64*192,3*3*192*384,4096,2048,${num_classes}" # SubFlow AlexNet for Cifar-10
fi

utilizations="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
# epochs="0 1 10 20"
epochs="0"

[[ -e logs/${dataset} ]] || mkdir -p logs/${dataset}

function process_subflow()
{
    subnet_id=$1
    use_class_importance=$2

    for utilization in $utilizations
    do
        for epoch in $epochs
        do
            echo "Computing subnetwork for utilization ${utilization} and epoch ${epoch}"
            # Create subnetwork
            python ${main_prog} -mode=sc -network_no=1 > logs/${dataset}/sub_network_${subnet_id}.log 2>&1

            # Train subnetwork with utilization and epoch
            if [[ ${epoch} -eq "0" ]]; then
                echo "Subnetwork ${subnet_id} training skipped for utilization $utilization, epoch $epoch"
                echo "" > logs/${dataset}/importance_${use_class_importance}_${iter}_${utilization}_${epoch}.log
            else
                echo "Subnetwork ${subnet_id} training for utilization $utilization, epoch $epoch"
                python ${main_prog} -mode=st -subflow_network_no=${subnet_id} -utilization=${utilization} -epochs=${epoch} -data=${dataset} > logs/${dataset}/importance_${use_class_importance}_${iter}_${utilization}_${epoch}.log 2>&1
            fi

            echo "Subnetwork ${subnet_id} inference for utilization $utilization, epoch $epoch"
            python ${main_prog} -mode=si -utilization=${utilization} -subflow_network_no=${subnet_id} -data=${dataset} -use_class_importance=${use_class_importance} >> logs/${dataset}/importance_${use_class_importance}_${iter}_${utilization}_${epoch}.log 2>&1
        done
    done
}


for (( iter = start_iter ; iter < total_iterations; ++iter))
do
    rm "${main_prog%.*}.obj"
    rm -r network*
    rm -r sub_network*

    subnet_id=1

    # First create a network and compute neuron importance.
    echo "======================================================"
    echo "Creating full network (${net_arch}: ${net_arch_conf}) for iteration ${iter}"
    python ${main_prog} -mode=c -layers="${net_arch_conf}" > logs/${dataset}/network.log 2>&1

    echo "Training full network"
    python ${main_prog} -mode=t  -data=${dataset} -network_no=1 -epochs=10 >> logs/${dataset}/network.log 2>&1

    echo "Full network inference"
    python ${main_prog} -mode=i  -data=${dataset} -network_no=1 >> logs/${dataset}/network.log 2>&1

    echo "Computing neuron importance metrics"
    python ${main_prog} -mode=ci  -data=${dataset} -network_no=1 >> logs/${dataset}/network.log 2>&1
    
    for use_class_importance in 0 1
    do
        process_subflow ${subnet_id} ${use_class_importance}
        subnet_id=$((subnet_id + 1))
    done
    
    echo "======================================================"
done
