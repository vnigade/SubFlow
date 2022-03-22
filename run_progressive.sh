#/bin/bash


export LD_LIBRARY_PATH=/cm/shared/apps/cuda10.0/toolkit/current/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

utilizations="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"
epochs="10"
subnet_id=1

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

echo "Computing subnetwork for utilization ${utilization} and epoch ${epoch}"
# Create subnetwork
python subflow.py -mode=sc -network_no=1 > /dev/null 2>&1

for utilization in $utilizations
do
    
    echo "Subnetwork ${subnet_id} training for utilization $utilization, epoch $epoch"
    python subflow.py -mode=st -subflow_network_no=${subnet_id} -utilization=${utilization} -epochs=${epochs} -data=mnist_data > /dev/null 2>&1
    

    echo "Subnetwork ${subnet_id} inference for utilization $utilization, epoch $epoch"
    python subflow.py -mode=si -utilization=${utilization} -subflow_network_no=${subnet_id} -data=mnist_data > logs/progressive_pre_${iter}_${utilization}_${epoch}.log 2> /dev/null        
done

for utilization in $utilizations
do
    echo "Subnetwork ${subnet_id} inference for utilization $utilization, epoch $epoch"
    python subflow.py -mode=si -utilization=${utilization} -subflow_network_no=${subnet_id} -data=mnist_data > logs/progressive_${iter}_${utilization}_${epoch}.log 2> /dev/null        
done

echo "======================================================"
