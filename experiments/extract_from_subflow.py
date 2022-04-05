"""
Loads a pre-trained SubFlow network and extracts its weights and biases.

todo: Currently only MNIST SubFlow networks have been tried.
"""
import argparse
import numpy as np
import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# todo: comment this in when running on DAS
# sub_conv2d_library = tf.load_op_library("./sub_conv2d.so")
# sub_matmul_library = tf.load_op_library("./sub_matmul.so")
# sub_conv2d_library = tf.load_op_library("/home/vinod/remote_files/das5/scratch/packages/SubFlow/sub_conv2d.so")
# sub_matmul_library = tf.load_op_library("/home/vinod/remote_files/das5/scratch/packages/SubFlow/sub_matmul.so")


# =================================================================================================
# Display
# =================================================================================================

def display_info(weight_ops, bias_ops, neuron_ops, output_ops) -> None:
    labels = ["Weights", "Bias", "Neuron", "Output"]
    groups = [weight_ops, bias_ops, neuron_ops, output_ops]
    for label, group in zip(labels, groups):
        print("=" * 100)
        print(label)
        print("=" * 100)
        for op in group:
            tensor = op.values()[0]
            layer_type = "Conv" if tensor.shape.rank == 4 else "Full"
            shape = tensor.shape
            fixed_shape = [v if v else 1 for v in shape.as_list()]
            float_count = np.prod(fixed_shape)
            print(f"{op.name}: {layer_type} {tensor.shape} ({float_count} Floats)")


def display_variables(variable_values) -> None:
    print("=" * 100)
    print("Trainable Variables")
    print("=" * 100)

    for variable, values in variable_values.items():
        print(f"{variable}: Shape {values.shape}")


# =================================================================================================
# Model
# =================================================================================================

def load_and_extract(base_network_folder: str, network_name: str, base_output_folder: str, layer_count: int = 6) -> None:
    network_folder = os.path.join(base_network_folder, network_name)
    network_file_path = os.path.join(network_folder, f"{network_name}")
    network_meta_file = network_file_path + ".meta"
    print(f"Loading meta file: {network_meta_file}")
    print(f"Restore from file path: {network_file_path}")

    saver = tf.compat.v1.train.import_meta_graph(network_meta_file)
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, network_file_path)

        # Extract trainable variables
        variables = tf.compat.v1.trainable_variables()
        variable_values = dict()
        for variable in variables:
            values = sess.run(variable)
            variable_values[variable.name] = values

        # Extract operation information
        operations = sess.graph.get_operations()
        operation_dict = {op.name: op for op in operations}

        weight_names = [f"weight_{i}" for i in range(layer_count) if f"weight_{i}" in operation_dict]
        bias_names = [f"bias_{i}" for i in range(layer_count) if f"bias_{i}" in operation_dict]
        neuron_names = [f"neuron_{i}" for i in range(layer_count) if f"neuron_{i}" in operation_dict]
        output_names = [f"output_{i}" for i in range(layer_count) if f"output_{i}" in operation_dict]

        weight_ops = [operation_dict[name] for name in weight_names]
        bias_ops = [operation_dict[name] for name in bias_names]
        neuron_ops = [operation_dict[name] for name in neuron_names]
        output_ops = [operation_dict[name] for name in output_names]

        # Display information
        display_info(weight_ops, bias_ops, neuron_ops, output_ops)
        display_variables(variable_values)

        # Write out weights and biases
        output_folder = os.path.join(base_output_folder, network_name)
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

        for name in weight_names + bias_names:
            variable_name = f"{name}:0"
            values = variable_values[variable_name]
            output_name = os.path.join(output_folder, name)
            np.save(output_name, values)


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--network_name", type=str, default="network1", help="The name of the network to load.")
    args.add_argument("--base_output_folder", type=str, default="../output/", help="The base output folder.")
    args.add_argument("--base_network_folder", type=str, default="../models/", help="The base network folder for input.")
    args.add_argument("--layer_count", type=int, default=6, help="The layer count of the network.")
    args = args.parse_args()

    # Load model and extract data
    load_and_extract(args.base_network_folder, args.network_name, args.base_output_folder, args.layer_count)


if __name__ == "__main__":
    main()
