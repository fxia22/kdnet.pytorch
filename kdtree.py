from collections import defaultdict
import scipy.spatial
import numpy as np


def get_cutdims(tree, max_depth=7):
    """
    Get balanced cut dimensions and indices from a scipy.spatial.KDTree.

    Args:
    - tree (scipy.spatial.ckdtree.cKDTree): a kdtree
    - max_depth (int): go to this depth on every node. If a branch is shorter or longer than max_depth then we repeat the end node or terminate the branch early.

    Returns:
    - cutdims (list): list , each item giving the split dimensions at that level
        - (array) numpy int64 array, giving which dimension the kdtree split along . Shape=(2**level,)
    - tree_idxs (list): list of numpy int64 arrays for each level.
        - (array) Each array is for one level and gives the indices of points at each node. Shape=(2**level, 2**(max_depth-level))
    """
    cutdims = defaultdict(list)
    tree_idxs = defaultdict(list)

    def _get_cutdims(tree, level=0, parent=None):
        if tree is None:
            # deal with premature leaf by repeating the leaf
            tree = parent

        if level >= max_depth:
            indices = tree.indices

            # make sure it's the right amount of indices for this depth
            n = 2**(max_depth - level)
            if len(indices) > n:
                # since we repeated the premature leafs we might get duplicate indices
                # or this might comes into play if the input is too large for the tree
                # print('crop', n, len(indices), level)
                inds = np.random.choice(range(len(indices)), n)
                indices = indices[inds]
            elif len(indices) < n:
                # pad if input is too small for tree
                # print('pad', n, len(indices), level)
                indices = np.concatenate([indices, indices[0:1].repeat(n - len(indices))])

            # end recursion
            tree_idxs[level].append(indices)
            return indices

        indices = np.concatenate([
            _get_cutdims(tree.lesser, level=level + 1, parent=tree),
            _get_cutdims(tree.greater, level=level + 1, parent=tree)
        ])
        if level < max_depth:
            tree_idxs[level].append(indices)

            # since we repeated premature leafs, we get invalid splits
            # in this case just use the parents
            split_dim = tree.split_dim
            if split_dim == -1:
                split_dim = parent.split_dim if (parent.split_dim > -1) else 0

            cutdims[level].append(split_dim)
            cutdims[level].append(split_dim)
        return indices

    # init the recursive search
    _get_cutdims(tree, level=0)

    # convert outputs
    tree_idxs = list(tree_idxs.values())
    cutdims = list(cutdims.values())

    # convert to numpy int64
    cutdims = [np.array(item).astype(np.int64) for item in cutdims]
    # also stack since they are constant sizes (for each level)
    tree_idxs = [np.stack(branch).astype(np.int64) for branch in tree_idxs]
    return cutdims, tree_idxs


def make_cKDTree(point_set, depth):
    """
    Take in a numpy pointset and quickly build a kdtree.

    Args:
    - point_set (numpy.array): array of points with shape=(rows, channels).
    - depth (int): tree depth

    Returns:
    - cutdims: (list) a list containing the dimension cut on each node on each level
    - tree: (list) the datapoints split into multiple arrays on each level
    - kdtree: (scipy.spatial.ckdtree.cKDTree)

    """
    kdtree = scipy.spatial.cKDTree(point_set, leafsize=1, balanced_tree=True)
    cutdims, tree_idxs = get_cutdims(kdtree.tree, max_depth=depth)

    # go from indices to points
    tree = [np.take(point_set, indices=indices, axis=0) for indices in tree_idxs]
    return cutdims, tree
