import numpy as np
from numba import njit

@njit
def erf_approx(x):
    # Constants for Chebyshev numerical approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0 if x >= 0 else -1.0
    abs_x = abs(x)
    t = 1.0 / (1.0 + p * abs_x)
    y = 1.0 - (
        ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-abs_x * abs_x)
    )
    return sign * y


@njit
def interp_periodic(x_val, x_first, dx, bias_array):
    n = len(bias_array)
    delta = x_val - x_first
    idx_left = int(np.floor(delta / dx))

    weight = (delta / dx) - idx_left

    idx_0 = idx_left % n
    idx_1 = (idx_left + 1) % n

    y0 = bias_array[idx_0]
    y1 = bias_array[idx_1]

    return y0 + (y1 - y0) * weight

@njit
def interp(x_val, x_first, dx, bias_array):
    idx = (x_val - x_first) / dx
    n = len(bias_array)

    if idx < 0:
        slope = (bias_array[1] - bias_array[0]) / dx
        return bias_array[0] + slope * (x_val - x_first)

    if idx >= n - 1:
        x_last = x_first + (n - 1) * dx
        slope = (bias_array[n - 1] - bias_array[n - 2]) / dx
        return bias_array[n - 1] + slope * (x_val - x_last)


    i = int(idx)
    x0 = x_first + i * dx
    frac = (x_val - x0) / dx

    return bias_array[i] + (bias_array[i + 1] - bias_array[i]) * frac



@njit
def hill_height(height, r_delta_T, bias_at_center):
    return height * np.exp(-r_delta_T * bias_at_center)



@njit
def gaussian_periodic(x, center, gauss_val, sigma, left, right, n_images, norm):
    inv_two_sigma_sq = -0.5 / (sigma * sigma)
    L = right - left
    n = len(x)

    k_values = np.arange(-n_images, n_images + 1)
    k_L = k_values * L
    
    for i in range(n):
        di = x[i] - center
        di -= L * np.floor(di / L + 0.5)
        
        accum = 0.0
        for j in range(len(k_L)):
            val = di + k_L[j]
            accum += np.exp(val * val * inv_two_sigma_sq)
            
        gauss_val[i] += acc * norm
    return gauss_val

@njit
def gaussian(x, center, gauss_val, sigma, kernel_mode):
    # sqrt(2 * pi) = 2.5066282746310002
    norm = 1.0 if kernel_mode != 1 else (1.0 / (2.5066282746310002 * sigma))
    inv_two_sigma_sq = -0.5 / (sigma * sigma)

    n = len(x)
    for i in range(n):
        dx_val = x[i] - center
        gauss_val[i] = np.exp(dx_val * dx_val * inv_two_sigma_sq) * norm

    return gauss_val


@njit
def sample_density(cell_mass, edges):
    rand_val = np.random.rand()

    current_cdf = 0.0
    n = len(cell_mass)
    chosen_idx = n - 1

    for i in range(n):
        current_cdf += cell_mass[i]
        if current_cdf >= rand_val:
            chosen_idx = i
            break

    rand_shift = np.random.rand()
    return edges[chosen_idx] + (edges[chosen_idx + 1] - edges[chosen_idx]) * rand_shift

@njit
def gaussian_integral_on_interval(center, sigma, kernel_mode, left, right):
    # sqrt(2) = 1.4142135623730951
    sqrt2_sigma = 1.4142135623730951 * sigma
    a = (left - center) / sqrt2_sigma
    b = (right - center) / sqrt2_sigma

    integral_normalized = 0.5 * (erf_approx(b) - erf_approx(a))

    if kernel_mode == 1:
        return integral_normalized
    else:
        # sqrt(2 * pi) = 2.5066282746310002
        return 2.5066282746310002 * sigma * integral_normalized


@njit
def gaussian_mean_on_interval(center, sigma, kernel_mode, left, right):
    """
    A(s) = 1/(right-left) * integral_left^right g(s-u) du
    """
    return gaussian_integral_on_interval( center, sigma, kernel_mode, left, right)/ (right - left)

@njit
def mcgovern_inverse_mean(x, inv_mean, sigma, kernel_mode, left,right):
    for i in range(len(x)):
        s = x[i]
        if s < left:
            s = left
        elif s > right:
            s = right
        mean_val = gaussian_mean_on_interval(s, sigma, kernel_mode, left, right)
        inv_mean[i] = 1.0 / mean_val
    return inv_mean

@njit
def gaussian_mcgovern_interval(x, center, gauss_val, inv_mean, sigma, kernel_mode, left, right):
    # sqrt(2 * pi) = 2.5066282746310002
    norm = (1.0 if kernel_mode != 1 else 1.0 / (2.5066282746310002 * sigma))
    inv_two_sigma_sq = -0.5 / (sigma * sigma)
    for i in range(len(x)):
        s = x[i]
        if s < left:
            s = left
        elif s > right:
            s = right
        ds = s - center
        raw_gaussian = (np.exp(ds * ds * inv_two_sigma_sq) * norm)
        gauss_val[i] = raw_gaussian * inv_mean[i]
    return gauss_val

@njit
def grid_mean_on_interval(values, x, dx, left, right):
    total = 0.0
    for i in range(len(x)):
        if x[i] > left and x[i] < right:
            total += values[i]
    return total * dx / (right - left)

@njit
def cell_mass_from_bias(V, F, log_rho, rho, cell_mass, beta, dx, alpha):
    n = len(V)

    max_log_rho = -alpha * V[0] - beta * F[0]
    log_rho[0] = max_log_rho
    for i in range(1, n):
        val = -alpha * V[i] - beta * F[i]
        log_rho[i] = val
        if val > max_log_rho:
            max_log_rho = val

    total_mass = 0.0
    for i in range(n):
        log_rho[i] -= max_log_rho
        rho_val = np.exp(log_rho[i])
        rho[i] = rho_val

        c_mass = rho_val * dx
        cell_mass[i] = c_mass
        total_mass += c_mass

    inv_total = 1.0 / total_mass
    for i in range(n):
        cell_mass[i] *= inv_total


@njit
def cell_mass_from_bias_interval(V, F, s_clamped, x, log_rho, rho, cell_mass, beta, dx, alpha):
    n = len(s_clamped)
    left = x[0]

    max_log_rho = -1e300

    for i in range(n):
        v_eff = interp(s_clamped[i], x[0], dx, V)
        val = -alpha * v_eff - beta * F[i]
        log_rho[i] = val
        if val > max_log_rho:
            max_log_rho = val

    total_mass = 0.0
    for i in range(n):
        log_rho[i] -= max_log_rho
        rho_val = np.exp(log_rho[i])
        rho[i] = rho_val
        c_mass = rho_val * dx
        cell_mass[i] = c_mass
        total_mass += c_mass

    inv_total = 1.0 / total_mass
    for i in range(n):
        cell_mass[i] *= inv_total

