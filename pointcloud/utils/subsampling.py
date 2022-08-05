import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    Code retrieved from: https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/datasets/common.py

    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points, sampleDl=sampleDl, verbose=verbose)
    elif labels is None:
        return cpp_subsampling.subsample(
            points, features=features, sampleDl=sampleDl, verbose=verbose
        )
    elif features is None:
        return cpp_subsampling.subsample(
            points, classes=labels, sampleDl=sampleDl, verbose=verbose
        )
    else:
        return cpp_subsampling.subsample(
            points,
            features=features,
            classes=labels,
            sampleDl=sampleDl,
            verbose=verbose,
        )
