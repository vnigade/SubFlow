#/bin/bash


export LD_LIBRARY_PATH=/cm/shared/apps/cuda10.0/toolkit/current/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

: ${random_iterations:=100}
: ${start_iter:=8}
utilizations="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
epochs="0 1 10 20"
subnet_id=1

for (( iter = start_iter ; iter < random_iterations; ++iter))
do
    rm subflow.obj
    rm -r network1
    rm -r sub_network*

    subnet_id=1

    # First create a network and compute random importance.
    echo "======================================================"
    echo "Creating full network for iteration ${iter}"
    python subflow.py -mode=c -layers='28*28*1,5*5*1*6,5*5*6*16,400,84,10' > /dev/null 2>&1

    echo "Training full network"
    python subflow.py -mode=t  -data=mnist_data -network_no=1 > /dev/null 2>&1

    echo "Computing neuron importance metrics"
    python subflow.py -mode=ci  -data=mnist_data -network_no=1 > /dev/null 2>&1

    
    for utilization in $utilizations
    do
        for epoch in $epochs
        do
            echo "Computing subnetwork for utilization ${utilization} and epoch ${epoch}"
            # Create subnetwork
            python subflow.py -mode=sc -network_no=1 > /dev/null 2>&1

            # Train subnetwork with utilization and epoch
            if [[ ${epoch} -eq "0" ]]; then
                echo "Subnetwork ${subnet_id} training skipped for utilization $utilization, epoch $epoch"
            else
                echo "Subnetwork ${subnet_id} training for utilization $utilization, epoch $epoch"
                python subflow.py -mode=st -subflow_network_no=${subnet_id} -utilization=${utilization} -epochs=${epoch} -data=mnist_data > /dev/null 2>&1
            fi

            echo "Subnetwork ${subnet_id} inference for utilization $utilization, epoch $epoch"
            python subflow.py -mode=si -utilization=${utilization} -subflow_network_no=${subnet_id} -data=mnist_data > logs/random_${iter}_${utilization}_${epoch}.log 2> /dev/null

            # Increment subnetwork id
            subnet_id=$((subnet_id + 1))
        done
    done
    echo "======================================================"
done
