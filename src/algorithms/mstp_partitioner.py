#!/usr/bin/env python3
"""
MSTP (Max-Subset τ-Partitioning) Algorithm

This module implements the MSTP algorithm for binary matrix partitioning.
The algorithm finds the maximum subset of columns that can be clustered
into at most tau clusters for a given binary matrix.
"""

import numpy as np


def MSTPPartition(X, tau, topN=np.inf):
    """
    MSTP (Max-Subset τ-Partitioning) algorithm for binary matrix clustering.
    
    This algorithm finds the maximum subset of columns that can be clustered
    into at most tau clusters for a given binary matrix.
    
    Args:
        X (np.ndarray): Binary matrix (rows × columns)
        tau (int): Maximum number of clusters allowed
        topN (int or float): Maximum number of partitions to keep at each iteration
                            (default: np.inf for no limit)
    Returns:
        np.ndarray: Solution list (tau × M) where each row indicates which 
                   columns to include for that tau value
    """
    
    def uniqueValMaker(X):
        _, vals1 = np.unique(X[:, 0], return_inverse=True)

        for a in range(1, X.shape[1]):
            vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)
            vals1 = (vals1 * vals2_unique.shape[0]) + vals2
            _, vals1 = np.unique(vals1, return_inverse=True)

        return vals1

    def preProcess(X):
        mean1 = np.mean(X, axis=1)
        argEven = np.argwhere(mean1 == 0.5)[:, 0]
        if argEven.shape[0] >= 1:
            mean1[argEven] = mean1[argEven] - 0.5 + X[argEven, 0]

        X[mean1 > 0.5] = 1 - X[mean1 > 0.5]

        inverse1 = uniqueValMaker(X)
        _, index1 = np.unique(inverse1, return_index=True)
        X = X[index1]

        return X, inverse1

    def slow_unique_partitions(partitionList):
        for a in range(partitionList.shape[0]):
            _, index1, inverse1 = np.unique(partitionList[a], return_index=True, return_inverse=True)
            _, index1 = np.unique(index1, return_inverse=True)
            inverse1 = index1[inverse1]
            partitionList[a] = inverse1

        inverse2 = uniqueValMaker(partitionList)
        
        return inverse2

    def get_unique_partitions(partitionList):
        # Normalizer
        argAll = np.argwhere(partitionList > -1)
        paster = np.zeros((partitionList.shape[0], tau * 2), dtype=int)
        paster[argAll[:, 0], partitionList[argAll[:, 0], argAll[:, 1]]] = argAll[:, 1]
        partitionList[argAll[:, 0], argAll[:, 1]] = paster[argAll[:, 0], partitionList[argAll[:, 0], argAll[:, 1]]]

        # Hash rows via view
        dtype = np.dtype((np.void, partitionList.dtype.itemsize * partitionList.shape[1]))
        partitionList = partitionList.view(dtype).ravel()

        _, inverse1 = np.unique(partitionList, return_inverse=True)

        return inverse1

    def debug_unique_partitions(partitionList):
        partition_inverse1 = get_unique_partitions(np.copy(partitionList))
        partition_inverse2 = slow_unique_partitions(partitionList)

        _, index1 = np.unique(partition_inverse1, return_index=True)
        _, index2 = np.unique(partition_inverse2, return_index=True)
        index1 = np.sort(index1)
        index2 = np.sort(index2)

        assert np.array_equal(index1, index2)

    def reform_partitions(partitionList):
        # Re-index partition sets 
        argAll = np.argwhere(partitionList > -1)
        paster = np.zeros((partitionList.shape[0], tau * 2), dtype=int)
        paster[argAll[:, 0], partitionList[argAll[:, 0], argAll[:, 1]]] = 1
        paster = (np.cumsum(paster, axis=1) * paster) - 1

        partitionList[argAll[:, 0], argAll[:, 1]] = paster[argAll[:, 0], partitionList[argAll[:, 0], argAll[:, 1]]]

        return partitionList

    def buildPartition(partitionList, X):
        # Find new partition sets
        allPairs = np.argwhere(np.zeros((partitionList.shape[0], X.shape[0]), dtype=int) > -1)
        partitionList_repeat = partitionList[allPairs[:, 0]]
        X_repeat = X[allPairs[:, 1]]
        partitionList_repeat = (partitionList_repeat * 2) + X_repeat

        partitionList_repeat = np.concatenate((partitionList, partitionList_repeat), axis=0)
        partition_inverse = get_unique_partitions(np.copy(partitionList_repeat))

        _, index1 = np.unique(partition_inverse, return_index=True)
        index1 = index1[index1 >= partitionList.shape[0]]  # This removes partitions that already previously existed

        partitionList_repeat = partitionList_repeat[index1]
        partitionList_repeat = reform_partitions(partitionList_repeat)

        return partitionList_repeat, partition_inverse

    def single_solver(partitionList, X, tau):
        partitionList_new, partition_inverse = buildPartition(partitionList, X)

        partitionSize = np.max(partitionList_new, axis=1) + 1
        partitionList_new = partitionList_new[partitionSize <= tau]

        return partitionList_new

    def findCompatable(partitionList, X):
        matchMatrix = np.zeros((partitionList.shape[0], X.shape[0]), dtype=int)
        allPairs = np.argwhere(matchMatrix > -1)
        partitionList_repeat = partitionList[allPairs[:, 0]]
        X_repeat = X[allPairs[:, 1]]

        argAll = np.argwhere(partitionList_repeat > -1)

        paster = np.zeros((partitionList_repeat.shape[0], tau, 2), dtype=int)
        paster[argAll[:, 0], partitionList_repeat[argAll[:, 0], argAll[:, 1]], X_repeat[argAll[:, 0], argAll[:, 1]]] = 1

        acrossBoth = np.max(np.sum(paster, axis=2), axis=1)
        allPairs_match = allPairs[acrossBoth <= 1]

        matchMatrix[allPairs_match[:, 0], allPairs_match[:, 1]] = 1

        return matchMatrix

    def findMaximalPartition(partitionList, X, X_inverse, tau):
        partitionSize = np.max(partitionList, axis=1) + 1

        _, count1 = np.unique(X_inverse, return_counts=True)

        matchMatrix = findCompatable(partitionList, X)

        # Objective: column selection only
        total_objective = np.sum(matchMatrix * count1.reshape((1, -1)), axis=1)

        solutionList = np.zeros((tau, X_inverse.shape[0]), dtype=int)

        for tau_now in range(1, tau + 1):
            argValid = np.argwhere(partitionSize <= tau_now)[:, 0]
            argBest = argValid[np.argmax(total_objective[argValid])]
            matchNow = matchMatrix[argBest, X_inverse]
            solutionList[tau_now - 1] = matchNow

        return solutionList

    def fullAlgorithm(X, tau, topN):
        X = X.T  # Easier to think through when transposed.

        X, X_inverse = preProcess(X)

        partitionList = np.zeros((1, X.shape[1]), dtype=int)
        partitionList_total = np.copy(partitionList)

        for iter in range(tau):
            if partitionList.shape[0] >= 1:
                partitionList = single_solver(partitionList, X, tau)

                if partitionList.shape[0] >= topN:
                    matchMatrix = findCompatable(partitionList, X)
                    numMatch = np.sum(matchMatrix, axis=1)
                    argTop = np.argsort(numMatch * -1)[:topN]
                    partitionList = partitionList[argTop]

                partitionList_total = np.concatenate((partitionList_total, np.copy(partitionList)), axis=0)

        solutionList = findMaximalPartition(partitionList_total, X, X_inverse, tau)

        return solutionList

    def verifySolution(solutionList, X, tau):
        X = X.T  # Easier to think through when transposed.

        for tau_now in range(1, tau + 1):
            solution1 = solutionList[tau_now - 1]
            X_now = X[solution1 == 1]
            if X_now.shape[0] >= 1:
                inverse1 = uniqueValMaker(X_now.T)
                print(np.max(inverse1) + 1, tau_now)
                assert np.max(inverse1) + 1 <= tau_now

    # Main algorithm execution
    try:
        solutionList = fullAlgorithm(np.copy(X), tau, topN)
        
        # Optional: verify the solution
        # verifySolution(solutionList, X, tau)
        
        return solutionList
        
    except Exception as e:
        import sys
        print(f"Error in MSTPPartition: {e}", file=sys.stderr)
        raise  # Re-raise the original exception


# Example usage function
def example_usage():
    """Example of how to use the MSTPPartition function."""
    
    # Create a test matrix
    X = np.array([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
    ])
    
    tau = 3
    topN = np.inf
    
    print("Input matrix:")
    print(X)
    print(f"Shape: {X.shape}")
    print(f"Tau: {tau}")
    print(f"TopN: {topN}")
    
    # Run the algorithm (original)
    solution = MSTPPartition(X, tau, topN)
    
    print(f"\nOriginal Solution:")
    print(solution)
    print(f"Solution shape: {solution.shape}")
    
    return solution


if __name__ == "__main__":
    # Run example when script is executed directly
    example_usage()

