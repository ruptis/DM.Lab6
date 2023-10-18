import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

DEFAULT_MATRIX = [
    [0., 5., 0.5, 2.],
    [5., 0., 1., 0.6],
    [0.5, 1., 0., 2.5],
    [2., 0.6, 2.5, 0.],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, help='Generate random matrix of given size')
    parser.add_argument('--method', type=str, default='min', help='Method of distance calculation (min, max)')
    args = parser.parse_args()

    if args.size is not None:
        matrix = np.random.random((args.size, args.size))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
    else:
        matrix = DEFAULT_MATRIX

    if args.method == 'max':
        matrix = [[1 / x if x != 0 else 0 for x in row] for row in matrix]

    print_matrix(matrix)
    linkage_matrix = linkage(squareform(matrix))
    dendrogram(linkage_matrix, labels=np.array(['x%d' % i for i in range(1, len(matrix) + 1)]))
    plt.show()


def print_matrix(matrix):
    print('\t' + '\t'.join(['x%d' % i for i in range(1, len(matrix) + 1)]))
    for i, row in enumerate(matrix):
        print('x%d\t' % (i + 1) + '\t'.join(['%.2f' % x for x in row]))


if __name__ == '__main__':
    main()
