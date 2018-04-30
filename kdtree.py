from collections import defaultdict
import scipy.spatial
import numpy as np


def get_cutdims(tree, max_depth=7):
    """Get cutdims from a scipy.spatial.KDTree."""
    cutdims = defaultdict(list)
    tree_idxs = defaultdict(list)

    def _get_cutdims(tree, level=0, parent=None):
        if not tree.lesser:
            if level < max_depth:
                tree = parent
            else:
                tree_idxs[level].append(tree.indices)
                return tree.indices

        indices = np.concatenate([
            _get_cutdims(tree.lesser, level=level + 1, parent=tree),
            _get_cutdims(tree.greater, level=level + 1, parent=tree)
        ])
        if level < max_depth:
            tree_idxs[level].append(indices)
            cutdims[level].append(tree.split_dim)
            cutdims[level].append(tree.split_dim)
        return indices

    # init
    _get_cutdims(tree, level=0)

    # post processes values
    tree_idxs = list(tree_idxs.values())
    print([len(tree_idxs[i]) for i in range(len(tree_idxs))])
    tree_idxs = [np.stack(tree_idxs[i]) for i in range(len(tree_idxs))]

    cutdims = list(cutdims.values())
    return cutdims, tree_idxs


def make_cKDTree(point_set, depth):
    """
    Take in a numpy pointset and quickly build a kdtree.

    Returns:
    - cutdims: (list) a list containing the dimension cut on each node on each level
    - tree: (list) the datapoints split into multiple arrays on each level

    """
    tree = scipy.spatial.cKDTree(point_set, leafsize=1, balanced_tree=True)
    cutdims, tree_idxs = get_cutdims(tree.tree, max_depth=depth)
    tree = [np.take(point_set, indices=indices, axis=0) for indices in tree_idxs]
    return cutdims, tree
