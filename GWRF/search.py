import numpy as np


def golden_section(a, c, delta, function, tol, max_iter, bw_max, int_score=False,
                   verbose=False):
    """
    Golden Section Search Algorithm.
    :param a: float Initial maximum search interval value.
    :param c: float Initial minimum search interval value.
    :param delta: float Constant used to determine the width of the search interval.
    :param function: function Objective function evaluated at different values.
    :param tol: float Tolerance to determine convergence.
    :param max_iter: integer Maximum number of iterations if not converged to tolerance.
    :param bw_max: float Maximum bandwidth value.
    :param int_score: boolean False for floating-point scores, True for integer scores.
    :param verbose: boolean Whether to print detailed information.
    :return:
        opt_val: float Optimal value.
        opt_score: float Optimal score of the kernel.
        output: list of tuples Search history.
    """
    # Calculate the values of b and d.
    b = a + delta * np.abs(c - a)
    d = c - delta * np.abs(c - a)

    opt_score = np.inf
    diff = 1.0e9
    iters = 0
    output = []
    dict1 = {}
    opt_val = np.inf
    # Loop while the difference is greater than the tolerance and the number of
    # iterations is less than the maximum and a is not infinity.
    while np.abs(diff) > tol and iters < max_iter and a != np.inf:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)

        if b in dict1:
            score_b = dict1[b]
        else:
            score_b = function(b)
            dict1[b] = score_b
            if verbose:
                print("Bandwidth: ", np.round(b, 2), ", score: ",
                      "{0:.2f}".format(score_b))

        if d in dict1:
            score_d = dict1[d]
        else:
            score_d = function(d)
            dict1[d] = score_d
            if verbose:
                print("Bandwidth: ", np.round(d, 2), ", score: ",
                      "{0:.2f}".format(score_d))
        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c - a)

        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c - a)

        output.append((opt_val, opt_score))

        opt_val = np.round(opt_val, 2)
        if (opt_val, opt_score) not in output:
            output.append((opt_val, opt_score))

        diff = score_b - score_d

    if a == np.inf or bw_max == np.inf:
        score_ols = function(np.inf)
        output.append((np.inf, score_ols))

        if score_ols <= opt_score:
            opt_score = score_ols
            opt_val = np.inf

        if verbose:
            print("Bandwidth: ", np.inf, ", score: ",
                  "{0:.2f}".format(score_ols[0]))
    return opt_val, opt_score, output


def equal_interval(l_bound, u_bound, interval, function, int_score=False,
                   verbose=False):
    """
    Equal Interval Search Algorithm, using interval as step size.
    :param l_bound: float Initial minimum search interval value.
    :param u_bound: float Initial maximum search interval value.
    :param interval: float Constant used to determine the width of the search interval.
    :param function: function Objective function evaluated at different values.
    :param int_score: boolean False for floating-point scores, True for integer scores.
    :param verbose: boolean Whether to print detailed information.
    :return:
        opt_val: float Optimal value.
        opt_score: float Optimal score of the kernel.
        output: list of tuples Search history.
    """
    def print_info(bandwidth, score):
        if verbose:
            print("Bandwidth:", bandwidth, ", score:", "{0:.2f}".format(score[0]))

    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a, 0)
        c = np.round(c, 0)
        b = np.round(b, 0)

    output = []

    score_a = function(a)
    print_info(a, score_a)
    output.append((a, score_a))

    opt_val = a
    opt_score = score_a

    while b < c:
        score_b = function(b)
        print_info(b, score_b)
        output.append((b, score_b))
        if score_b < opt_score:
            opt_val = b
            opt_score = score_b
        b = b + interval

    score_c = function(c)
    print_info(c, score_c)
    output.append((c, score_c))

    if score_c < opt_score:
        opt_val = c
        opt_score = score_c

    return opt_val, opt_score, output
