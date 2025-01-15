import numpy as np


class Kernel(object):
    def __init__(self, i, points, kdtree, bw=None, fixed=True, function='triangular', coords_i=None,
                 eps=1.0000001):
        """
        Initialize an instance of the Kernel class.
        :param i: The index of the data point.
        :param points: The array of data points.
        :param kdtree: The kd-tree object used to find nearest neighbors.
        :param bw: Bandwidth, default is None.
        :param fixed: Indicates whether to use a fixed bandwidth, default is True.
        :param function: The type of kernel function, default is 'triangular'.
        :param coords_i: Optional coordinates, default is None.
        :param eps: Factor used to calculate adaptive bandwidth, default is 1.0000001.
        """

        # Use kdtree to obtain the distances and indices of the nearest neighbors.
        if coords_i is None:
            distances, indices = kdtree.kneighbors(points[i].reshape(1, -1))
        else:
            distances, indices = kdtree.kneighbors(coords_i.reshape(1, -1))

        # Get the distance and index of the nearest neighbor.
        k_distance = distances[0]
        k_indices = indices[0]

        self.function = function.lower()

        # Calculate the bandwidth.
        if fixed:
            self.bandwidth = float(bw)
        else:
            self.bandwidth = k_distance[int(bw) - 1] * eps

        points_nums = len(points)
        self.dvec = np.zeros(points_nums)
        self.dvec[k_indices] = k_distance  # Vectorized assignment of distances

        # for j in range(points_nums):
        #     self.dvec[k_indices[j]] = k_distance[j]

        # Calculate the corresponding weight value based on different kernel function types.
        self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)

        if self.function == "bisquare":
            self.kernel[(self.dvec >= self.bandwidth)] = 0

    def _kernel_funcs(self, zs):
        if self.function == 'triangular':  # 三角核函数
            return 1 - zs
        elif self.function == 'uniform':  # 均匀核函数
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':  # 二次核函数
            return (3. / 4) * (1 - zs**2)
        elif self.function == 'quartic':  # 四次核函数
            return (15. / 16) * (1 - zs**2)**2
        elif self.function == 'gaussian':  # 高斯核函数
            return np.exp(-0.5 * (zs)**2)
        elif self.function == 'bisquare':  # 双平方核函数
            return (1 - (zs)**2)**2
        elif self.function == 'exponential':  # 指数核函数
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)
