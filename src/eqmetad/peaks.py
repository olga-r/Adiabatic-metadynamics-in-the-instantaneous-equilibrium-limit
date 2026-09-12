import numpy as np
from scipy.signal import find_peaks
from eqmetad.utils import smoothen_log_density

def detect_peaks(dx, sigma, peak_threshold, filter_width, p):
    smooth_logp = smoothen_log_density(p, filter_width)
    peak_indices, _ = find_peaks( smooth_logp, prominence=peak_threshold, distance=3 )
    peak_indices = list(map(int, peak_indices))
    boundary_window = max(5, int(round(sigma / dx)))
    if smooth_logp[0] > smooth_logp[1]:
        left_peak = smooth_logp[0] - np.min(
            smooth_logp[1:boundary_window + 1]
        )
        if left_peak >= peak_threshold:
            peak_indices.append(0)
    if  smooth_logp[-1] > smooth_logp[-2]:
        right_peak = smooth_logp[-1] - np.min(
            smooth_logp[-boundary_window - 1:-1]
        )
        if right_peak >= peak_threshold:
            peak_indices.append(len(smooth_logp) - 1)
    return sorted(peak_indices)

def analyze_peaks(peaks_indices, p, grid_centers, masses, left, right):
    L = right-left
    max_peak_std = 0.01*L
    boundaries = [0]
    for left_peak, right_peak in zip(peaks_indices[:-1], peaks_indices[1:]):
        local_min = left_peak + int(np.argmin(p[left_peak:right_peak + 1]))
        boundaries.append(local_min + 1 )
    boundaries.append(len(p))
    peaks_data = {}
    peaks_mass = 0
    for j, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        local_mass = masses[a:b]
        m = float(np.sum(local_mass))
        local_x = grid_centers[a:b]
        center = float(np.dot(local_x, local_mass) / m)
        variance = float(np.dot((local_x - center) ** 2, local_mass) / m)
        std = np.sqrt(max(variance, 0.0))
        if std < max_peak_std*0.5:
            peaks_mass += m
        if std < max_peak_std:
            if a == 0:
                boundary_second_moment = float(np.dot((local_x-left) ** 2, local_mass) / m)
                boundary_rms = np.sqrt(boundary_second_moment)
            elif b == len(p):
                boundary_second_moment = float(
                    np.dot((right - local_x) ** 2, local_mass) / m
                )
                boundary_rms = np.sqrt(boundary_second_moment)
            else:
                boundary_rms = np.nan
            peaks_data[int(j)] = { "mass": m, "center": center, "variance": variance, "std": std, "boundary_rms": boundary_rms}
    return peaks_mass, peaks_data
