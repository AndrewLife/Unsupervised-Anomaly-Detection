#!/usr/bin/env python3
#
# umi.py
# 
# This file contains a reference implementation of our method of intelligently classifying
# data using morisita indices. There is also a sample program that makes use of our
# classification function. Although our sample uses a fixed, known-size dataset, our
# classification function is also suitable for use in streams of unknown length.

from collections import deque

UNCLASSIFIED   = 0
NORMAL         = 1
ANOMALY        = 2
PENDING        = 3
FALSE_POSITIVE = 4
FALSE_NEGATIVE = 5
TRUE_POSITIVE  = 6
TRUE_NEGATIVE  = 7

def lce(data, min_cluster=2, num_benchmarks=15):
    # Number of dimensions per datapoint
    feature_count = len(data[0])

    # Size of a quadrat (halves itself each trial)
    delta = 1

    lce = np.zeros(num_benchmarks, dtype=int)
    cells = []

    for benchmark in range(num_benchmarks):
        #  Number of quadrats per dimension. Same as 1 / delta
        q = np.power(2, benchmark)

        # Total number of quadrats
        Q = np.power(q, feature_count)

        # The domains of each cell
        # Eg: benchmark = 2, delta = 0.25
        # [0, 0.25, 0.5, 0.75, 1.0]

        # quadrat_sums contains number of datapoints in each quadrat.
        # Recall that Q = q ^ feature_count
        quadrat_sums = np.zeros((q,) * feature_count)

        for i in range(len(data)):
            quadrat = tuple(int(np.ceil(data[i][f] / delta)) - 1 if data[i][f] != 0 else 0 for f in range(feature_count))
            quadrat_sums[quadrat] += 1

            if i == len(data) - 1:
                cells.append(quadrat)

        for i in np.ndenumerate(quadrat_sums):
            quadrat_sum = quadrat_sums[i[0]]

            if quadrat_sum < min_cluster:
                continue

            product = 1
            for j in range(min_cluster):
                product *= quadrat_sum - j
            lce[benchmark] += product
        
        delta /= 2

    return lce, cells

def classify(datapoint, trusted_data, last_classification, last_cells=None):
    """Classifies data as NORMAL, ANOMALY, or PENDING.

    Args:
        datapoint: The current datapoint to classify. May be array-like for multi-dimensional data.
        trusted_data: A double-ended queque of datapoints that are trusted.
        last_classification: The classification of the previous datapoint.
        last_cells: A list containing the cell indicies that the last point landed in.
    Returns:
        A tuple containing the revised classification for the last datapoint, and the classification for the new datapoint, and the list of the quadrat indices that the last point landed in.
    """
    
    revision = last_classification
    classification = UNCLASSIFIED
    cells = None

    # Number of dimensions per datapoint
    feature_count = len(datapoint)

    # Check if the datapoint is an exact duplicate of a trusted datapoint
    if datapoint in trusted_data:
        if last_classification == PENDING:
            revision = NORMAL
        classification = NORMAL

    # Check if the data lands in the same cluster as a trusted datapoint
    else:
        all_data = np.array([*np.unique(trusted_data, axis=0), datapoint], dtype=float, copy=True)

        # Minimum and maximum value for each dimension
        feature_mins = np.amin(all_data, axis=0)
        feature_maxs = np.amax(all_data, axis=0)

        # Normalize the data to the range [0.0, 1.0]
        for feature in range(feature_count):
            for i in range(len(all_data)):
                all_data[i][feature] = (all_data[i][feature] - feature_mins[feature]) / (feature_maxs[feature] - feature_mins[feature])

        # Minimum cluster size
        min_cluster = 2

        # Number of times to divide quadrat size
        num_benchmarks = 15

        # Find LCE for trusted set and trusted set + new data
        all_lce, cells = lce(all_data, min_cluster, num_benchmarks)
        trusted_lce = lce(all_data[:-1], min_cluster, num_benchmarks)[0]

        all_lce_val = 0
        trusted_lce_val = 0

        cell = None
        last_cell = None

        # If all_lce_val > trusted_lce_val, the new data resides in a pre-existing cluster
        for i in range(len(all_lce)):
            lce_val = all_lce[i]
            if lce_val > 0:
                all_lce_val = lce_val
                if last_cells != None:
                    cell = cells[i]
                    last_cell = last_cells[i]
        for lce_val in trusted_lce:
            if lce_val > 0:
                trusted_lce_val = lce_val

        shared_cell = last_cell == cell
        clustered = all_lce_val > trusted_lce_val

        if clustered:
            classification = NORMAL

            if last_classification == PENDING:
                if shared_cell:
                    revision = NORMAL
                else:
                    print(".")
                    revision = ANOMALY

        else:
            classification = PENDING

            if last_classification == PENDING:
                revision = ANOMALY

    # If the new data is NORMAL, the replace the oldest trusted data with the new data
    if classification == NORMAL:
        trusted_data.popleft()

    # If pending data was normal, remove the oldest datapoint that we didn't remove last time
    if last_classification == PENDING and revision == NORMAL:
        trusted_data.popleft()

    # If the pending data was anomalous, remove it from the trusted dataset
    if revision == ANOMALY:
        trusted_data.pop()

    # Keep track of current datapoint for future reference
    trusted_data.append(datapoint)

    return revision, classification, cells


if __name__ == '__main__':
    import numpy as np
    import argparse
    import json

    DATA_PATH = './nab/realTweets/realTweets/Twitter_volume_%s.csv'

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sample of LCE method for detecting anomalies in twitter dataset')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
    parser.add_argument('-t', '--trusted-size', dest='trusted_size', type=int, help='The size of the subset of trusted data')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    dataset = args.dataset
    trusted_size = args.trusted_size
    verbose = args.verbose
    
    # Load specified dataset into memory.
    all_data = np.loadtxt(open(DATA_PATH % (dataset), 'rb'), delimiter=",", skiprows=1, usecols=1, dtype=int)
    timestamps = np.loadtxt(open(DATA_PATH % (dataset), 'rb'), delimiter=",", skiprows=1, usecols=0, dtype=str)
    data_size = len(all_data)
    
    # Create results buffer.
    # The trusted data are marked as normal.
    # Remaining data is still unclassified.
    results = np.zeros(data_size, dtype=int)
    results[:trusted_size] = np.full(trusted_size, NORMAL, dtype=int)
    
    # Create the trusted buffer
    # This buffer contains the current set of data that are known to be trustworthy.
    # At the moment, it just contains the first `trusted_size` datapoints.
    # We add space for two additional indices for pending data which is temporarily trusted.
    trusted_data = deque(maxlen=trusted_size + 2)
    for i in range(0, trusted_size): trusted_data.append((all_data[i],))

    anomalies = 0

    # Classify the remaining data
    last_cells = None
    for i in range(trusted_size, data_size):
        # The new, unclassified data
        datapoint = (all_data[i],)

        # The result of the last classification. For the first result, this is NORMAL
        last_classification = results[i - 1]

        # Get a revision for the last classification if it was pending, and a classification for the new data.
        revision, fresh, last_cells = classify(datapoint, trusted_data, last_classification, last_cells)

        # Store our results
        results[i - 1] = revision
        results[i] = fresh

        if verbose:
            print("Point %d classified as %d" % (i-1, revision))

        if revision == ANOMALY:
            anomalies += 1

    print("Classified %d anomalies out of %d points" % (anomalies, data_size - trusted_size))
    
    fh = open("labels/combined_labels.json")
    labels_raw = fh.read()
    fh.close()
    anomalies = json.loads(labels_raw)["realTweets/Twitter_volume_%s.csv" % (dataset)]

    true_pos, true_neg, false_pos, false_neg, = 0, 0, 0, 0
    for i in range(len(results)):
        if timestamps[i] in anomalies:
            if results[i] == ANOMALY:
                results[i] = TRUE_POSITIVE
                true_pos += 1
            else:
                results[i] = FALSE_NEGATIVE
                false_neg += 1
        else:
            if results[i] == NORMAL:
                results[i] = TRUE_NEGATIVE
                true_neg += 1
            else:
                results[i] = FALSE_POSITIVE
                false_pos += 1

    print("True Positives:", true_pos)
    print("True Negatives:", true_neg)
    print("False Positives:", false_pos)
    print("False Negatives:", false_neg)

