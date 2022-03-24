"""
Fits polygons to the weight functions.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from network import NetworkCollection
from typing import List


# =================================================================================================
# Main
# =================================================================================================

def polyfit(collection: NetworkCollection, degree: int = 4) -> None:
    # Combine all the network weights in one big array (#weights x #networks)
    combined_weights = collection.combined_weights

    # Compute the difference relative to the first network.
    # The assumption is here that the first one represents full utilization (1.0) and the following are decreasing utilizations in order.
    weight_differences = np.diff(combined_weights, axis=1)
    weight_count, difference_count = weight_differences.shape

    # Fit polynomial
    x = np.arange(collection.count, dtype=np.float32)
    y = combined_weights[1, :]

    coefficients, (residuals, _, _, _) = np.polynomial.polynomial.polyfit(x, y, degree, full=True)
    polynomial = np.polynomial.polynomial.Polynomial(coefficients)
    print(f"Polynomial: {polynomial}")
    print(f"Residuals: {residuals}")
    y_fit = polynomial(x)

    # Display original data points and fitted polynomial
    plt.scatter(x, y)
    plt.plot(x, y_fit)
    plt.show()


def polyfit_all(collection: NetworkCollection, output_file: str, degree: int = 4) -> List[np.polynomial.polynomial.Polynomial]:
    # Combine all the network weights in one big array (#weights x #networks)
    combined_weights = collection.combined_weights

    # Compute the difference relative to the first network.
    # The assumption is here that the first one represents full utilization (1.0) and the following are decreasing utilization in order.
    weight_differences = np.diff(combined_weights, axis=1)
    weight_count, _ = weight_differences.shape

    # Fit polynomials
    x = np.arange(collection.count, dtype=np.float32)
    polynomials = list()
    for i in range(weight_count):
        y = combined_weights[i, :]
        coefficients, (residuals, _, _, _) = np.polynomial.polynomial.polyfit(x, y, degree, full=True)
        polynomial = np.polynomial.polynomial.Polynomial(coefficients)
        polynomials.append(polynomial)

    # Save results to file
    file = open(output_file, "wb")
    pickle.dump(polynomials, file)

    return polynomials


def polyfit_load(input_file: str) -> List[np.polynomial.polynomial.Polynomial]:
    file = open(input_file, "rb")
    polynomials = pickle.load(file)
    return polynomials


def evaluate_fit(collection: NetworkCollection, polynomials: List[np.polynomial.polynomial.Polynomial]) -> None:
    combined_weights = collection.combined_weights
    fitted_weights = np.zeros(combined_weights.shape, dtype=np.float32)
    weight_count, _ = combined_weights.shape
    assert len(polynomials) == weight_count

    # Predict weights using polynomial fit
    x = np.arange(collection.count, dtype=np.float32)
    for i, polynomial in enumerate(polynomials):
        y_fit = polynomial(x)
        fitted_weights[i, :] = y_fit

    # Compute average error
    minimum_weight_value = np.min(combined_weights)
    maximum_weight_value = np.max(combined_weights)
    average_weight_value = np.mean(combined_weights)

    differences = combined_weights - fitted_weights
    absolute_differences = np.abs(differences)
    squared_differences = differences * differences
    per_weight_differences = np.sum(differences, axis=1)
    per_weight_squared_differences = np.sum(squared_differences, axis=1)
    per_weight_absolute_differences = np.sum(absolute_differences, axis=1)

    mean_per_weight_difference = np.mean(per_weight_differences, axis=0)
    mean_per_weight_squared_difference = np.mean(per_weight_squared_differences, axis=0)
    mean_per_weight_absolute_difference = np.mean(per_weight_absolute_differences, axis=0)

    # Display errors
    print(f"Minimum weight value: {minimum_weight_value}")
    print(f"Maximum weight value: {maximum_weight_value}")
    print(f"Average weight value: {average_weight_value}")
    print(f"Mean difference: {mean_per_weight_difference}")
    print(f"Mean squared difference: {mean_per_weight_squared_difference}")
    print(f"Mean absolute difference: {mean_per_weight_absolute_difference}")


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--collection_file", type=str, default="../output/collection.pickle", help="The network collection to load.")
    args = args.parse_args()

    # Load network collection
    collection = NetworkCollection.load_from_file(args.collection_file)
    print(collection)

    # Fit polynomials to weight functions
    degree = 3
    temp_file = f"polynomials_{degree}.pickle"

    if not os.path.exists(temp_file):
        # polyfit(collection)
        print(f"Computing new polynomials (degree={degree}), saving to file {temp_file}.")
        polynomials = polyfit_all(collection, temp_file, degree)
    else:
        print(f"Loading polynomials (degree={degree}) from file {temp_file}.")
        polynomials = polyfit_load(temp_file)

    evaluate_fit(collection, polynomials)


if __name__ == "__main__":
    main()
