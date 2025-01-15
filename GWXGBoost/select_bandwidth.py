import warnings
import numpy as np
from scipy.optimize import minimize_scalar
from tools import create_kdtree
from GWXGBoost import GWXGBoost
from search import golden_section, equal_interval
from diagnostics import calculate_cv_value


class SelectBandwidth(object):
    def __init__(self, coords, feature, target, n_estimators=100, max_depth=None, bandwidth=1.0, kernel='bisquare',
                 criterion='mse', fixed=False, spherical=False, n_jobs=-1, random_state=None):
        """
        Initialize the Select_Bandwidth class.
        :param coords: Coordinates data.
        :param feature: Feature data.
        :param target: Target data.
        :param n_estimators: Number of estimators, default is 100.
        :param max_depth: Maximum depth, default is None.
        :param bandwidth: Bandwidth, default is 1.0.
        :param kernel: Kernel type, default is 'bisquare'.
        :param criterion: Criterion, default is'mse'.
        :param fixed: Whether the bandwidth is fixed, default is False.
        :param spherical: Whether to use spherical coordinates, default is False.
        :param n_jobs: Number of jobs, default is -1.
        :param random_state: Random state, default is None.
        """

        self.coords = coords
        self.X_ = feature
        self.Y_ = target
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bandwidth = bandwidth
        self.criterion = criterion
        self.fixed = fixed
        self.spherical = spherical
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.search_params = {}
        self.search_method = None
        self.int_score = False
        self.tol = None
        self.max_iter = None
        self.verbose = False
        self.interval = None
        self.bw_min = None
        self.bw_max = None
        self.rss_score = False
        self.bws_same_times = None
        self.sel_hist = None

    def search(self, search_method='golden_section',
               bw_min=None, bw_max=None, interval=0.0, tol=1.0e-6,
               max_iter=200, rss_score=False, bws_same_times=5,
               verbose=False, pool=None):
        """
        Perform the bandwidth search.
        :param search_method: Search method, default is 'golden_section'.
        :param bw_min: Minimum bandwidth, default is None.
        :param bw_max: Maximum bandwidth, default is None.
        :param interval: Interval for search, default is 0.0.
        :param tol: Tolerance, default is 1.0e-6.
        :param max_iter: Maximum number of iterations, default is 200.
        :param rss_score: Whether to use RSS score, default is False.
        :param bws_same_times: Number of times bandwidths remain the same, default is 5.
        :param verbose: Whether to print verbose information, default is False.
        :param pool: Pool for parallelization, default is None.
        :return: The selected bandwidth.
        """
        self.search_method = search_method
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        if pool:
            warnings.warn(
                "The pool parameter is no longer used and will have no effect; parallelization is default and "
                "implemented using joblib instead.",
                RuntimeWarning, stacklevel=2)

        self.rss_score = rss_score
        self.bws_same_times = bws_same_times
        self.search_params['search_method'] = search_method
        self.search_params['bw_min'] = bw_min
        self.search_params['bw_max'] = bw_max
        self.search_params['interval'] = interval
        self.search_params['tol'] = tol
        self.search_params['max_iter'] = max_iter

        self.int_score = not self.fixed

        self._bw()
        self.sel_hist = self.bw[-1]
        return self.bw[0]

    def _create_gwr_func(self):
        def gwr_func(bw):
            return calculate_cv_value(GWXGBoost(coords=self.coords, feature=self.X_, target=self.Y_,
                                           n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           bandwidth=bw, kernel=self.kernel, criterion=self.criterion,
                                           fixed=self.fixed, spherical=self.spherical, n_jobs=self.n_jobs,
                                           random_state=self.random_state).fit())

        return gwr_func

    def _bw(self):
        """
        Perform the actual bandwidth selection process.
        """
        print('Searching for bandwidth...')
        self.gwr_func = self._create_gwr_func()

        if self.search_method == 'golden_section':
            a, c = self._init_section(feature=self.X_, coords=self.coords)
            delta = 0.38197  # 1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, self.gwr_func, self.tol,
                                     self.max_iter, self.bw_max, self.int_score,
                                     self.verbose)
        elif self.search_method == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                                     self.gwr_func, self.int_score, self.verbose)
        elif self.search_method == 'scipy':
            self.bw_min, self.bw_max = self._init_section(feature=self.X_, coords=self.coords)
            if self.bw_min == self.bw_max:
                raise Exception(
                    'Maximum bandwidth and minimum bandwidth must be distinct for scipy optimizer.'
                )
            self._optimize_result = minimize_scalar(
                self.gwr_func, bounds=(self.bw_min, self.bw_max), method='bounded')
            self.bw = [self._optimize_result.x, self._optimize_result.fun, []]
        else:
            raise TypeError('Unsupported computational search method ',
                            self.search_method)

    def _init_section(self, feature, coords):
        """
         Initialize the search section.
         :param feature: Feature data.
         :param coords: Coordinate data.
         :return: Initial values for the search section.
         """
        n_vars = feature.shape[1]
        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            kd_tree = create_kdtree(coords, n_neighbors=n, spherical=self.spherical)
            dists, _ = kd_tree.kneighbors(coords)
            min_min_dist = np.min(dists[:, 1])
            max_max_dist = np.max(dists[:, -1])
            a = min_min_dist / 2.0
            c = max_max_dist * 2.0

        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None and self.bw_max is not np.inf:
            c = self.bw_max

        return a, c
