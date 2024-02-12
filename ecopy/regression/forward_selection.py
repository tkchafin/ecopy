import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


def forward_sel_par(Y, X, method, alpha=0.05, K=None, R2thresh=0.99,
                    R2more=0.001, adjR2thresh=0.99, Yscale=False,
                    verbose=True):
    """
    Perform Forward Selection with Parametric Testing on Multivariate Data

    This function implements a parametric forward selection procedure for
    selecting explanatory variables in regression models and Redundancy
    Analysis (RDA), inspired by the forward.sel.par function from the
    Adespatial R package. It is designed to handle both univariate and
    multivariate response data, employing an F-test as described by Miller
    and Farr (1971) for multivariate cases.

    The selection process stops when specified criteria are met, including the
    maximum number of variables (K), R-square thresholds (R2thresh and
    adjR2thresh), a minimum improvement in R-square (R2more), and a
    significance level (alpha).

    Parameters
    ----------
    Y : numpy.ndarray or pandas.DataFrame
        Response data matrix with n rows and m columns, containing
        quantitative variables.
    X : numpy.ndarray or pandas.DataFrame
        Explanatory data matrix with n rows and p columns, containing
        quantitative variables.
    method : callable
        A function or class implementing the regression model, which accepts Y
        and X as arguments and returns an object with an R2 attribute.
    alpha : float, optional
        Significance level for stopping the selection process if the p-value of
        a variable exceeds this threshold. Default is 0.05.
    K : int, optional
        Maximum number of variables to be selected. Defaults to one less than
        the number of rows in X.
    R2thresh : float, optional
        R-square threshold to stop the forward selection process. Values can
        range from 0.001 to 1. Default is 0.99.
    adjR2thresh : float, optional
        Adjusted R-square threshold to stop the forward selection process. Any
        value smaller than 1 is acceptable. Default is 0.99.
    R2more : float, optional
        The minimum improvement in R-square required to continue the selection
        process. Default is 0.001.
    Yscale : bool, optional
        Indicates whether to standardize the variables in Y to variance 1.
        Automatically set to True if Y is multivariate. Default is False.
    verbose : bool, optional
        If True, additional diagnostics are printed during the selection
        process. Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the selected variables along with their
        selection order, R-square values, cumulative R-square values,
        cumulative adjusted R-square values, F-statistics, and p-values.

    References
    ----------
    Dray, S., Blanchet, G., Borcard, D., Guenard, G., Jombart, T., Larocque,
    G., Legendre, P., Madi, N., Wagner, H.H., & Dray, M.S. (2018).
    Package ‘adespatial’. R package version 2018, pp.3-8.

    Miller, J. K., & Farr, S. D. (1971). Bimultivariate redundancy: A
    comprehensive measure of interbattery relationship. Multivariate
    Behavioral Research, 6, 313–324.

    Example
    -------
    import numpy as np
    import pandas as pd
    import ecopy as ep

    # Example data
    Y = pd.DataFrame(np.random.randn(100, 3),
            columns=['Resp1', 'Resp2', 'Resp3'])
    X = pd.DataFrame(np.random.randn(100, 5),
            columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])

    # Perform forward selection
    result = ep.forward_sel_par(Y, X, method=ep.rda(), alpha=0.05,
                verbose=True)
    print(result)
    """

    if K is None:
        K = X.shape[1] - 1
    n, p = Y.shape

    # Standardize Y if multivariate
    if Yscale or p > 1:
        Y = StandardScaler().fit_transform(Y)
        if verbose:
            print("The variables in response matrix Y have been standardized")

    selected_vars, results, R2cum = [], [], 0
    stop_reason = ""

    for _ in range(K):
        best_increase, best_var, best_model_stats = 0, None, None

        for var in X.columns.difference(selected_vars):
            current_vars = selected_vars + [var]
            model_result = method(Y, X[current_vars])
            R2 = get_R2_value(model_result)
            increase = R2 - R2cum

            if increase > best_increase:
                best_increase, best_var, best_model_stats = increase, var, (
                    R2, model_result)

        if best_var:
            R2prev, R2cum, _ = R2cum, *best_model_stats
            Fstat, pval = FPval(R2cum, R2prev, n, len(current_vars), p)
            adjR2 = RsquareAdj(R2cum, n, len(current_vars), p)

            if pval <= alpha and \
                    adjR2 <= adjR2thresh and \
                    R2cum <= R2thresh and \
                    best_increase > R2more:
                selected_vars.append(best_var)
                results.append({'variable': best_var, 'R2': R2cum,
                    'adjR2': adjR2, 'F': Fstat, 'pval': pval})
                if verbose:
                    print(
                        f"Selected {best_var}: R2cum={R2cum:.4f}, ",
                        f"adjR2={adjR2:.4f}, F={Fstat:.4f}, pval={pval:.4g}")
            else:
                stop_reason = "Criteria met" if pval > alpha else \
                    "No significant improvement"
                break
        else:
            stop_reason = "No variable improves the model"
            break

    if verbose and stop_reason:
        print(f"Selection stopped: {stop_reason}")

    return pd.DataFrame(results)


def FPval(R2cum, R2prev, n, m, p):
    """
    Compute the partial F-statistic and associated p-value.
    """
    df1 = p
    df2 = (n - m - 1) * p
    Fstat = (R2cum - R2prev) / (1 - R2cum) * df2 / df1
    pval = stats.f.sf(Fstat, df1, df2)
    return Fstat, pval


def RsquareAdj(R2, n, m, p):
    """
    Calculate adjusted R-square.
    """
    return 1 - (1 - R2) * (n - 1) / (n - m - 1)


def get_R2_value(model_result):
    """
    Retrieve the R2 value from the model result, supporting both attributes
    and callable methods.
    """
    R2 = getattr(model_result, 'R2', None)
    if R2 is None:
        raise AttributeError(
            "The model result does not have an 'R2' attribute or method.")

    if callable(R2):
        return R2()
    else:
        return R2

