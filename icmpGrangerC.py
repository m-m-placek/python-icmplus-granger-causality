# Michal M. Placek, last modified 2021-12-20, version 1.0
#
# This file is desinged to be run by ICM+ software (Cambridge Enterprise, Cambridge, UK, https://icmplus.neurosurg.cam.ac.uk).
# See the following paper about running Python code from ICM+.
# M.M. Placek, A. Khellaf, B.L. Thiemann, M. Cabeleira, P. Smielewski, "Python-Embedded Plugin Implementation in ICM+:
# Novel Tools for Neuromonitoring Time Series Analysis with Examples Using CENTER-TBI Datasets", in: B. Depreitere,
# G. Meyfroidt, F. Güiza (Eds.), Intracranial Pressure and Neuromonitoring XVII. Acta Neurochirurgica, Supplementum,
# Springer, Cham, 2021: pp. 255–260. https://doi.org/10.1007/978-3-030-59436-7_48
# Manuscript available here: https://www.repository.cam.ac.uk/handle/1810/321847
#
# Future work: rewrite the code in such a way that VAR model fitting and GC estimation is done only once when ICM+ calls
# function more times with the same parameters just to get different output variables, e.g. the magnitude and p-value.

from gct_MP import grangercausalitytests_MP, var_specrad, is_pos_def, calculate_gc_expF_from_input, gc_pval, \
    fill_missing_multichannel_data, is_missing_data_limit_satisfied, integration_order
from statsmodels.tsa.api import VAR
import numpy as np
import math
from scipy import stats


class GrangerC:

    # static variables
    tests = {
        'SSRF'  : 'ssr_ftest',
        'SSRX2' : 'ssr_chi2test',
        'LR'    : 'lrtest',
        'PAR'   : 'params_ftest',
        'MVGCF' : 'mvgc_F',
        'MVGCX2': 'mvgc_chi2'
    }
    fill_methods = {
        'NOISE': 'noise',
        'NEAR' : 'nearest',
        'LIN'  : 'linear'
    }

    # DO NOT MODIFY THIS METHOD. It is a part of the ICM+--Python interface.
    def set_parameter(self, param_name, param_value):
        setattr(self, param_name, param_value)

    # ICM+--Python plugin will set values of these parameters for us.
    def __init__(self):
        self.sampling_freq = None
        self.testType      = None
        self.outParam      = None
        self.maxOrder      = None
        self.infoCrit      = None
        self.fill          = None
        self.forceDiff     = None
        self.statioChk     = None
        self.maxIntegO     = None
        self.rhoThresh     = None
        self.MDLim         = None
        self.maxGapLim     = None
        self.MDT           = None

    def __del__(self):
        pass

    # 'calculate' is the main work-horse function.
    # It is called with a data buffer (one or more) of size corresponding to the Calculation Window.
    # It must return one floating-point number.
    # It take the following parameters:
    # sig1 - input variable/signal 1,
    # sig2 - input variable/signal 2,
    # ts_time - part of the data time stamp - number of milliseconds since midnight,
    # ts_date - Part of the data time stamp - One plus number of days since 1/1/0001.
    # It can also use the data sampling frequency self.sampling_freq
    # and all the variables already set at the initialisation time via ICM+--Python plugin.
    def calculate(self, sig1,sig2, ts_time,ts_date):

        X = np.column_stack((sig2, sig1))  # 2nd_column -> 1st_column, i.e. sig1 -> sig2 causality

        if self.MDT == 'IGN':  # missing data handling
            if self.maxGapLim is None:
                self.maxGapLim = np.Inf
            if self.MDLim is None:
                self.MDLim = np.Inf
            # first check if missing data limits are satisfied
            if is_missing_data_limit_satisfied(X, gap_limit=self.maxGapLim, perc_limit=self.MDLim):
                fill_method = GrangerC.fill_methods[self.fill]
                X = fill_missing_multichannel_data(X, alter_all_channels=True, fill_method=fill_method, standardise=True)
                # fill_missing_multichannel_data changes data in place when fill_method='noise'
            else:
                return np.nan
        else:
            X = stats.zscore(X, axis=0)
            # It may be wise to return NaN when missing data treatment in ICM+ was set to different option than 'IGN' (ignore)

        # if we're interested only in p-value, then we can use the implementation: .fit().test_causality()
        # instead of grangercausalitytests() function
        if self.infoCrit == 'NONE' or self.maxOrder == 1:
            order = self.maxOrder
        else:
            mdl = VAR(X)
            # mdl_fit = mdl.fit(maxlags=self.maxOrder, ic=self.infoCrit.lower())
            lagOrderResults = mdl.select_order(maxlags=self.maxOrder)  # trend='nc' option is problematic here
            order = int( getattr(lagOrderResults, self.infoCrit.lower()) )
            # order number in the lagOrderResults object is stored as <class 'numpy.int32'> and it's better to cast it
            # to <class 'int'> because when numpy.int32 is returned to ICM+, it's not converted properly
            if self.outParam == 'ORDER':
                return order
            order = max(order, 1)  # to prevent situation when optimal order is zero (Python's implementation allows that)
        if self.outParam == 'ORDER':
            return order

        if self.outParam in ('SPRAD', 'POSDEF', 'INTEG'):
            if self.outParam == 'INTEG':
                integ_ord, _ = integration_order(X, lags_order=order)
                return integ_ord
            if self.forceDiff:
                X = np.diff(X, n=self.maxIntegO, axis=0)
            mdl = VAR(X)
            var_fit_result = mdl.fit(maxlags=order, trend='nc', method='ols')
            if self.outParam == 'SPRAD':
                A = var_fit_result.coefs  # corresponds to A in Matlab (but dimensions are arranged in different way)
                return var_specrad(A)
            elif self.outParam == 'POSDEF':
                SIG = var_fit_result.sigma_u_mle  # is similar to SIG in Matlab
                return int( is_pos_def(SIG) )   # <class 'bool'> isn't converted properly when returned to ICM+, thus casting to <class 'int'>
            else:
                return np.NaN

        # Old implementation of grangercausalitytests() function was performing analysis for all 1,2,...,maxlag orders.
        # New implementation can avoid this by passing array/list as the 'maxlag' argument.
        # Alternatively, syntax .fit().test_causality() can also be used.
        if self.testType in ('SSRF', 'SSRX2', 'LR', 'PAR'):

            if self.forceDiff:
                X = np.diff(X, n=self.maxIntegO, axis=0)
            elif self.statioChk:
                _, data_verified = integration_order(X, lags_order=order, max_integ_ord=self.maxIntegO,
                                                     rho_threshold=self.rhoThresh, trend='nc')
                if data_verified is None:  # data cannot be stationarised by differencing
                    return np.NaN
                else:
                    X = data_verified[0]
                    # X = stats.zscore(X, axis=0)  # forcing sigma=1 was more important when filling the data with noise

            gct_result = grangercausalitytests_MP(X, [order], addconst=False, verbose=False)
            # In the original implementation of grangercausalitytests function, addconst must be True
            # The implementation modified by me allows for addconst=False.

            # gct_result[order][0]['testType'] - statstics
            # gct_result[order][1][ [0,1,2] ] - the OLS estimation results for the restricted model, the unrestricted
            #                                   model and the restriction (contrast) matrix for the parameter f_test.

            # some AIC and BIC are available in gct_result[order][1][ [0,1] ].aic  .bic
            # (but this can be something different than AIC/BIC from VAR model)

            statistic = gct_result[order][0][ GrangerC.tests[self.testType] ]

            if self.outParam == 'PVAL':
                return statistic[1]
            elif self.outParam == 'STAT': # This is the value of F or chi2 statistics.
                return statistic[0]       # p-value is derived from this number and corresponding degrees of freedom.
            elif self.outParam == 'MAG':
                regressions = gct_result[order][1]  # Magnitude is independent of the self.testType
                ssr_own   = regressions[0].ssr
                ssr_joint = regressions[1].ssr
                magnitude = math.log(ssr_own / ssr_joint)
                return magnitude
            else:
                return np.NaN

        elif self.testType in ('MVGCF', 'MVGCX2'):
            # when VAR is unstable or nearly unstable estimation is very slow

            # mdl = VAR(X)
            # var_fit_result = mdl.fit(maxlags=order, trend='nc', method='ols')
            # A = var_fit_result.coefs  # corresponds to A in Matlab (but dimensions are arranged in different way)
            # if var_specrad(A) > 0.99:
            #    return np.NaN

            # MVGC requires order np.column_stack((x1, x2)), instead of gct Python functions np.column_stack((x2, x1))
            X = X[:, ::-1]  # swap columns

            if self.forceDiff:
                X = np.diff(X, n=self.maxIntegO, axis=0)
            do_statio_check = self.statioChk and not self.forceDiff

            expF = calculate_gc_expF_from_input(X, order, ac_dec_tol=1E-8, ac_lags_limit=2000,
                                stationarise=do_statio_check, max_integ_ord=self.maxIntegO, rho_threshold=self.rhoThresh)
            expF12 = float(expF[1, 0])  # numpy types aren't converted properly when returned to ICM+, thus casting to float
            if self.outParam == 'PVAL':
                nobs = X.shape[0]
                statistic = 'F' if self.testType == 'MVGCF' else 'chi2'
                return gc_pval(expF12, nobs=nobs, order=order, statistic=statistic)
            elif self.outParam == 'MAG':
                return math.log(expF12)
            else:       # self.outParam == 'STAT' is not implemented for MVGC, it's probably unnecessary
                return np.NaN
        else:
            return np.NaN
