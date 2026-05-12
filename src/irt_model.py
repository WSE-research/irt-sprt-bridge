"""1PL Rasch IRT model via Joint Maximum Likelihood Estimation (JMLE)."""

import numpy as np


def fit_rasch_jmle(
    X: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> dict:
    """Fit a 1PL (Rasch) IRT model via JMLE.

    Args:
        X: Binary agreement matrix (n_items × n_models). 1=agree, 0=disagree.
           NaN entries are treated as missing and excluded.
        max_iter: Maximum JMLE iterations.
        tol: Convergence tolerance on max parameter change.

    Returns:
        dict with keys: theta (model abilities), difficulty (item difficulties),
        n_iter, converged, model_fit (log-likelihood, AIC, BIC).
    """
    n_items, n_models = X.shape
    mask = ~np.isnan(X)
    X_clean = np.where(mask, X, 0)

    eps = 1e-8
    model_sums = np.nansum(X, axis=0)
    model_counts = mask.sum(axis=0)
    model_rates = np.clip(model_sums / np.maximum(model_counts, 1), eps, 1 - eps)

    item_sums = np.nansum(X, axis=1)
    item_counts = mask.sum(axis=1)
    item_rates = np.clip(item_sums / np.maximum(item_counts, 1), eps, 1 - eps)

    theta = np.log(model_rates / (1 - model_rates))
    b = -np.log(item_rates / (1 - item_rates))

    converged = False
    for iteration in range(max_iter):
        theta_old = theta.copy()
        b_old = b.copy()

        logit_p = theta[np.newaxis, :] - b[:, np.newaxis]
        logit_p = np.clip(logit_p, -30, 30)
        P = 1 / (1 + np.exp(-logit_p))

        P_masked = P * mask

        for j in range(n_models):
            r_j = X_clean[:, j][mask[:, j]].sum()
            e_j = P_masked[:, j].sum()
            info_j = (P_masked[:, j] * (1 - P[:, j]) * mask[:, j]).sum()
            if info_j > eps:
                theta[j] += (r_j - e_j) / info_j

        for i in range(n_items):
            s_i = X_clean[i, :][mask[i, :]].sum()
            e_i = P_masked[i, :].sum()
            info_i = (P_masked[i, :] * (1 - P[i, :]) * mask[i, :]).sum()
            if info_i > eps:
                b[i] -= (s_i - e_i) / info_i

        b -= b.mean()

        delta = max(np.max(np.abs(theta - theta_old)), np.max(np.abs(b - b_old)))
        if delta < tol:
            converged = True
            break

    logit_p = theta[np.newaxis, :] - b[:, np.newaxis]
    logit_p = np.clip(logit_p, -30, 30)
    P = 1 / (1 + np.exp(-logit_p))
    ll = np.sum(
        (X_clean * np.log(P + eps) + (1 - X_clean) * np.log(1 - P + eps)) * mask
    )
    n_obs = mask.sum()
    n_params = n_models + n_items - 1
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n_obs)

    return {
        "theta": theta,
        "difficulty": b,
        "n_iter": iteration + 1,
        "converged": converged,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "n_obs": int(n_obs),
        "n_items": n_items,
        "n_models": n_models,
    }


def item_fit_infit(X: np.ndarray, theta: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute infit mean-square for each item (Wright & Masters)."""
    mask = ~np.isnan(X)
    X_clean = np.where(mask, X, 0)

    logit_p = theta[np.newaxis, :] - b[:, np.newaxis]
    logit_p = np.clip(logit_p, -30, 30)
    P = 1 / (1 + np.exp(-logit_p))
    W = P * (1 - P)

    residual_sq = ((X_clean - P) ** 2) * mask
    weighted_resid = (residual_sq * mask).sum(axis=1)
    expected_var = (W * mask).sum(axis=1)

    infit = weighted_resid / np.maximum(expected_var, 1e-8)
    return infit


def person_fit_infit(X: np.ndarray, theta: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute infit mean-square for each person/model."""
    mask = ~np.isnan(X)
    X_clean = np.where(mask, X, 0)

    logit_p = theta[np.newaxis, :] - b[:, np.newaxis]
    logit_p = np.clip(logit_p, -30, 30)
    P = 1 / (1 + np.exp(-logit_p))
    W = P * (1 - P)

    residual_sq = ((X_clean - P) ** 2) * mask
    weighted_resid = (residual_sq * mask).sum(axis=0)
    expected_var = (W * mask).sum(axis=0)

    infit = weighted_resid / np.maximum(expected_var, 1e-8)
    return infit
