# Michal M. Placek, last modified 2021-12-20, version 1.0

import numpy as np
from scipy import stats
from numpy.random import default_rng
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.tools.validation import (array_like, string_like, bool_like,
                                          int_like, dict_like, float_like)
from statsmodels.tsa.tsatools import lagmat, lagmat2ds, add_trend
import pandas as pd
from statsmodels.tsa.vector_ar import util as VAR_util
import math


# My modified version of statsmodels.tsa.stattools.grangercausalitytests (v0.11.1) allowing addConst=False option.
# This small modification led to obtaining practically the same results as the GCCA MATLAB toolbox.
def grangercausalitytests_MP(x, maxlag, addconst=True, verbose=True):
    """
    Four tests for granger non causality of 2 time series.

    All four tests give similar results. `params_ftest` and `ssr_ftest` are
    equivalent based on F test which is identical to lmtest:grangertest in R.

    Parameters
    ----------
    x : array_like
        The data for test whether the time series in the second column Granger
        causes the time series in the first column. Missing values are not
        supported.
    maxlag : {int, Iterable[int]}
        If an integer, computes the test for all lags up to maxlag. If an
        iterable, computes the tests only for the lags in maxlag.
    addconst : bool
        Include a constant in the model.
    verbose : bool
        Print results.

    Returns
    -------
    dict
        All test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        test statistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.

    Notes
    -----
    TODO: convert to class and attach results properly

    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

    'params_ftest', 'ssr_ftest' are based on F distribution

    'ssr_chi2test', 'lrtest' are based on chi-square distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Granger_causality

    .. [2] Greene: Econometric Analysis

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.stattools import grangercausalitytests
    >>> import numpy as np
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> data = data.data[['realgdp', 'realcons']].pct_change().dropna()

    # All lags up to 4
    >>> gc_res = grangercausalitytests(data, 4)

    # Only lag 4
    >>> gc_res = grangercausalitytests(data, [4])
    """
    x = array_like(x, 'x', ndim=2)
    if not np.isfinite(x).all():
        raise ValueError('x contains NaN or inf values.')
    addconst = bool_like(addconst, 'addconst')
    verbose = bool_like(verbose, 'verbose')
    try:
        lags = np.array([int(lag) for lag in maxlag])
        maxlag = lags.max()
        if lags.min() <= 0 or lags.size == 0:
            raise ValueError('maxlag must be a non-empty list containing only '
                             'positive integers')
    except Exception:
        maxlag = int_like(maxlag, 'maxlag')
        if maxlag <= 0:
            raise ValueError('maxlag must a a positive integer')
        lags = np.arange(1, maxlag + 1)

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError("Insufficient observations. Maximum allowable "
                         "lag is {0}".format(int((x.shape[0] - int(addconst)) /
                                                 3) - 1))

    resli = {}

    for mlg in lags:
        result = {}
        if verbose:
            print('\nGranger Causality')
            print('number of lags (no zero)', mlg)
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        # add constant
        if addconst:
            dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
            rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
                                   np.eye(mxlg, mxlg),
                                   np.zeros((mxlg, 1))))
        else:
            # raise NotImplementedError('Not Implemented')
            dtaown = dta[:, 1:(mxlg+1)]  # previous commented code changed to +1
            dtajoint = dta[:, 1:]
            rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
                                   np.eye(mxlg, mxlg)))

        # Run ols on both models without and with lags of second variable
        res2down   = OLS(dta[:, 0], dtaown  ).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # print results
        # for ssr based tests see:
        # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        # the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = ((res2down.ssr - res2djoint.ssr) /
                res2djoint.ssr / mxlg * res2djoint.df_resid)
        if verbose:
            print('ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                  ' df_num=%d' % (fgc1,
                                  stats.f.sf(fgc1, mxlg,
                                             res2djoint.df_resid),
                                  res2djoint.df_resid, mxlg))
        result['ssr_ftest'] = (fgc1,
                               stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                               res2djoint.df_resid, mxlg)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        if verbose:
            print('ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, '
                  'df=%d' % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg))
        result['ssr_chi2test'] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        # likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        if verbose:
            print('likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %
                  (lr, stats.chi2.sf(lr, mxlg), mxlg))
        result['lrtest'] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        #rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
        #                           np.eye(mxlg, mxlg),
        #                           np.zeros((mxlg, 1))))
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print('parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                  ' df_num=%d' % (ftres.fvalue, ftres.pvalue, ftres.df_denom,
                                  ftres.df_num))
        result['params_ftest'] = (np.squeeze(ftres.fvalue)[()],
                                  np.squeeze(ftres.pvalue)[()],
                                  ftres.df_denom, ftres.df_num)

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli


#def calc_gc_magnitude(gc_regressions):
#    ssr_own   = gc_regressions[0].ssr
#    ssr_joint = gc_regressions[1].ssr
#    magnitude = math.log(ssr_own / ssr_joint)
#    return magnitude


def var_specrad(A_coefs):
    """Calculate the spectral radius of VAR model
    Based on utils\var_specrad.m from MVGC toolbox v1.0
    Michal Placek, 2020-05-07"""
    #if A_coefs.ndim != 3:
    #    raise Exception('Input argument must be 3-dimensional array representing VAR model')
    A_coefs = array_like(A_coefs, None, ndim=3)
    # (n, n1, p) = A_coefs.shape  # dimensions order in Matlab
    (p, n, n1) = A_coefs.shape
    if n != n1:
        raise ValueError('VAR coefficients matrix has bad shape')
    # p1n = (p-1) * n
    # A1 = A_coefs.transpose((1, 0, 2)).reshape((n, n*p), order='C') # C means C-like (rows), not columns!
    # A1 = np.block([ [A1                             ],
    #                 [np.eye(p1n), np.zeros((p1n, n))]  ])
    A1 = VAR_util.comp_matrix(A_coefs)
    eigenvalues, _ = np.linalg.eig(A1)
    #print(A1)
    rho = max(abs(eigenvalues))
    return rho
# In Python, there is is_stable method!
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VARProcess.is_stable.html#statsmodels.tsa.vector_ar.var_model.VARProcess.is_stable


def integration_order(X, lags_order, max_integ_ord=10, rho_threshold=1.0, trend='nc'):
    # how many differencing should be done to make data covariance-stationary
    # If 'trend' is 'no constant', then it's assumed that user provides demeaned X.
    if lags_order <= 0:
        raise ValueError("'lags_order' must be a positive integer")
    if max_integ_ord < 0:
        raise ValueError("'max_integ_ord' must be a non-negative integer")
    if not (0. < rho_threshold <= 1.):
        raise ValueError("'rho_threshold' must be in the interval (0, 1]")
    integ_ord = 0
    while True:
        mdl = VAR(X)
        var_fit_result = mdl.fit(maxlags=lags_order, trend=trend, method='ols')
        # var_fit_result.is_stable() is equivalent to var_specrad(var_fit_result.coefs) < 1
        rho = var_specrad(var_fit_result.coefs)
        if rho < rho_threshold:
            return integ_ord, (X, rho, var_fit_result)
        elif integ_ord >= max_integ_ord:
            return 99999, None
        else:
            integ_ord += 1
            X = np.diff(X, axis=0)  # diff shortents data segment
            if trend in ('nc', 'n'):
                X -= X.mean(axis=0)


def mrdivide_real(B, A):
    # Matlab-equivalent of mrdivide function (for real functions only)
    # https://stackoverflow.com/a/59380798
    # B/A === np.linalg.solve(A.conj().T, B.conj().T).conj().T # conj required when dealing with complex numbers
    return np.linalg.solve(A.T, B.T).T


def acf_to_var(G):
    # Calculate VAR parameters from autocovariance sequence.
    # Equivalent to autocov_to_var (Whittle's recursive LWR algorithm) from MVGC toolbox v1.0 in Matlab.
    G = array_like(G, None, ndim=3)
    (q1, n, n1) = G.shape  # in Matlab q1 is in the last (3rd) dimension
    if n != n1:
        raise ValueError('vector autocovariance ACF coefficients matrix has bad shape')
    q = q1 - 1
    G0 = G[0, :, :]                                                                    # covariance
    GF = G[1:, :, :].transpose((1,2,0)).reshape((n, q*n), order='F').transpose()       # forward  autocov sequence
    GB = np.flip(G[1:, :, :], axis=0).transpose((1,0,2)).reshape((q*n, n), order='F')  # backward autocov sequence

    AF = np.zeros((n, q*n))  # forward  coefficients
    AB = np.zeros((n, q*n))  # backward coefficients (reversed compared with Whittle's treatment)

    # initialise recursion
    k = 1  # model order
    r = q - k
    kf = range(k*n)       # forward  indices
    kb = range(r*n, q*n)  # backward indices
    AF[:, kf] = mrdivide_real(GB[kb, :], G0)
    AB[:, kb] = mrdivide_real(GF[kf, :], G0)

    for k in range(2, q+1):
        # In Python, matrix multiplication is performed by np.matmul(,) function or equivalently @ operator
        # Operator * performs element-wise multiplication (like .* in Matlab)
        AAF = mrdivide_real(GB[(r-1)*n : r*n, :] - AF[:, kf] @ GB[kb, :],  G0 - AB[:, kb] @ GB[kb, :])  # DF / VB
        AAB = mrdivide_real(GF[(k-1)*n : k*n, :] - AB[:, kb] @ GF[kf, :],  G0 - AF[:, kf] @ GF[kf, :])  # DB / VF
        AFprev = AF[:, kf]
        ABprev = AB[:, kb]
        r = q - k
        kf = range(k*n)
        kb = range(r*n, q*n)
        AF[:, kf] = np.hstack((AFprev - AAF @ ABprev,  AAF))
        AB[:, kb] = np.hstack((AAB,  ABprev - AAB @ AFprev))

    SIG = G0 - AF @ GF
    # Python dimensions convension (p,n,n) instead of Matlab's (n,n,p)
    A_reconstructed = AF.reshape((n,n,q), order='F').transpose((2,0,1))
    return A_reconstructed, SIG


def acf_to_gc(G):
    # Calculate time-domain Granger causality from autocovariance sequence
    # Similar to autocov_to_pwcgc from MVGC v1.0 toolbox in Matlab.
    G = array_like(G, None, ndim=3)
    (_, n, n1) = G.shape
    if n != n1:
        raise ValueError('vector autocovariance ACF coefficients matrix has bad shape')
    F = np.empty((n,n))
    F.fill(np.nan)
    # full regression
    _, SIG = acf_to_var(G)
    SIG_diag = SIG.diagonal()

    for j in range(n):
        jo = list(range(n))
        jo.remove(j)
        # reduced regression(s)
        _, SIGj = acf_to_var(G[:, jo, jo])
        SIGj_diag = SIGj.diagonal()
        for ii in range(n-1):
            i = jo[ii]
            F[i, j] = SIGj_diag[ii] / SIG_diag[i]

    return F


def gc_pval(expF, nobs, order, statistic='F'):
    if statistic not in ('F', 'chi2'):
        raise ValueError("'statistic' must be 'F' or 'chi2'")
    expF_array = array_like(expF, None, ndim=2)
    if expF_array.shape[0] > 2 or expF_array.shape[1] > 2:
        raise NotImplementedError('Only 2 variable GC is supported')
    m = nobs - order  # effective number of observations (p-lag autoregression loses p observations per trial)
    nx = 1  # number of target ("to") variables
    ny = 1  # number of source ("from") variables
    if statistic == 'F':
        # nx must be 1 here
        nz = 0  # number of conditioning variables
        df = m - order * (1 + ny + nz)
        fgc1 = (expF - 1) * df / order
        pval = stats.f.sf(fgc1, order, df) # very close to Matlab's mvgc_pval(...,'F')
    else:  # 'chi2'
        d = order * nx * ny
        fgc2 = m * (expF - 1)
        pval = stats.chi2.sf(fgc2, d)  # similar (but not very close) to mvgc_pval(...,'chi2')
    return pval


def calculate_gc_expF_from_input(X, order, ac_dec_tol=1E-8, ac_lags_limit=10000, stationarise=False,
                                 max_integ_ord=3, rho_threshold=1.):
    if not isinstance(stationarise, bool):
        raise ValueError("'stationarise' must be boolean.")
    X -= X.mean(axis=0)
    if stationarise:
        _, data_verified = integration_order(X, lags_order=order, max_integ_ord=max_integ_ord,
                                             rho_threshold=rho_threshold, trend='nc')
        if data_verified is None:  # data cannot be stationarised by differencing
            channels_count = X.shape[1]
            return np.fill((channels_count, channels_count), np.NaN)
        else:
            (X, rho, var_fit_result) = data_verified
            #X = stats.zscore(X, axis=0)  # forcing sigma=1 was more important when filling the data with noise
    else:
        mdl = VAR(X)
        var_fit_result = mdl.fit(maxlags=order, trend='nc', method='ols')
        rho = var_specrad(var_fit_result.coefs)  # for checking if specRad < 1, is_stable method can be used
    ac_dec_lags = math.ceil(math.log(ac_dec_tol) / math.log(rho))
    ac_lags = min(ac_dec_lags, ac_lags_limit)
    # G = var_fit_result.acf(ac_lags)
    G = var_acf(var_fit_result.coefs, var_fit_result.sigma_u_mle, nlags=ac_lags)
    # sigma_u and sigma_u_mle are not equal, but F_mag calculated using these two different sigmas are equal!!!
    # From Magnitude or exp(Magnitude), we can easily go to 'fgc1' ratio in gct function and obtain F/chi2 as well as p-values
    expF = acf_to_gc(G)
    return expF


def is_pos_def(matrix):
    # maybe, we should also check if matrix is symmetric
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def find_longest_streak(sequence):
    """In the sequence of bools, find the longest subsequence of True values and return its length.
    sequence - sequence of bools"""
    curr_streak = longest_streak = 0
    for elem in sequence:
        if elem:
            curr_streak += 1
            longest_streak = max(longest_streak, curr_streak)
        else:
            curr_streak = 0
    return longest_streak


def find_longest_gap(sig):
    return find_longest_streak( np.logical_not(np.isfinite(sig)) )


def invalid_data_percentage(sig):
    return 100 * sum( np.logical_not(np.isfinite(sig)) ) / len(sig)


def mutual_missing(X):
    if X.ndim > 2:
        raise ValueError('Input should be a matrix.')
    are_pairs_inval = np.any( np.logical_not(np.isfinite(X)) , axis=1)   # Matlab dim=2
    inval_count = sum(are_pairs_inval)
    total_count = len(are_pairs_inval)
    inval_perc = 100 * inval_count / total_count
    return find_longest_streak(are_pairs_inval), inval_perc
    # return both ??


def fill_missing_multichannel_data(data, alter_all_channels, fill_method='noise', standardise=False): ## inplace=False
    if not isinstance(alter_all_channels, bool):
        raise ValueError("'alter_all_channels' must be boolean.")
    if isinstance(data, tuple):
        data = np.column_stack(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError('Invalid type of data. It should be np.ndarray or tuple of arrays/lists.')
    if fill_method not in ('noise', 'linear', 'nearest'):
        raise ValueError("Unrecognised value of 'fill_method'")
    if not isinstance(standardise, bool):
        raise ValueError("'standardise' must be boolean.")

    are_points_inval = np.logical_not(np.isfinite(data))
    if alter_all_channels:
        are_rows_inval = np.any(are_points_inval, axis=1, keepdims=True)
        data[are_rows_inval.flatten(), :] = np.nan
        channels_count = data.shape[1]
        are_points_inval = np.tile(are_rows_inval, (1, channels_count))
    else:
        data[are_points_inval] = np.nan
    if standardise:
        data = stats.zscore(data, axis=0, nan_policy='omit')
    if fill_method == 'noise':
        rng = default_rng()
        data[are_points_inval] = rng.standard_normal( np.sum(are_points_inval) )  # assuming that data are standarised (mu=0, sigma=1)
        # otherwise, they should be adjusted for mu and sigma
        return data
    else:
        data_frame = pd.DataFrame(data)
        # interpolate(method='linear', limit_direction='both') will do linear interpolation where possible and nearest at the edges
        data_frame.interpolate(method=fill_method, limit_direction='both', inplace=True)
        return data_frame.to_numpy()


def is_missing_data_limit_satisfied(data, gap_limit, perc_limit):
    if isinstance(data, tuple):     # code copied from fill_missing_multichannel_data; consider extracting new function
        data = np.column_stack(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError('Invalid type of data. It should be np.ndarray or tuple of arrays/lists.')
    longest_gap, inval_perc = mutual_missing(data)
    return longest_gap <= gap_limit and inval_perc <= perc_limit
