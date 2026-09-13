import numpy as np
from scipy.signal import find_peaks
from eqmetad.utils import smoothen_log_density

def detect_peaks(dx, sigma, peak_threshold, filter_width, p, periodic=False):
    smooth_logp = smoothen_log_density(p, filter_width)
    N = len(smooth_logp)
    boundary_window = max(5, int(round(sigma / dx)))

    if periodic:
        padded_logp = np.concatenate([
            smooth_logp[-boundary_window:],
            smooth_logp,
            smooth_logp[:boundary_window]
        ])

        peak_indices, _ = find_peaks(padded_logp, prominence=peak_threshold, distance=3)

        actual_peaks = set()
        for idx in peak_indices:
            orig_idx = (idx - boundary_window) % N
            actual_peaks.add(int(orig_idx))
        return sorted(list(actual_peaks))

    else:
        peak_indices, _ = find_peaks(smooth_logp, prominence=peak_threshold, distance=3)
        peak_indices = list(map(int, peak_indices))
        if smooth_logp[0] > smooth_logp[1]:
            left_peak = smooth_logp[0] - np.min(smooth_logp[1:boundary_window + 1])
            if left_peak >= peak_threshold:
                peak_indices.append(0)
        if smooth_logp[-1] > smooth_logp[-2]:
            right_peak = smooth_logp[-1] - np.min(smooth_logp[-boundary_window - 1:-1])
            if right_peak >= peak_threshold:
                peak_indices.append(N - 1)
        return sorted(peak_indices)



def analyze_peaks(peaks_indices, p, grid_centers, masses, left, right, periodic=False):
    L = right - left
    max_peak_std = 0.01 * L
    N = len(p)

    if not peaks_indices:
        return 0.0, {}

    if periodic:
        boundaries = []
        extended_peaks = peaks_indices + [peaks_indices[0] + N]
        for idx1, idx2 in zip(extended_peaks[:-1], extended_peaks[1:]):
            indices = np.arange(idx1, idx2 + 1) % N
            local_min_rel = int(np.argmin(p[indices]))
            boundaries.append(indices[local_min_rel])

        boundaries = sorted(list(set(boundaries)))
        if len(boundaries) == 1:
            segments = [(boundaries[0], boundaries[0])]
        else:
            segments = list(zip(boundaries[:-1], boundaries[1:])) + [(boundaries[-1], boundaries[0])]
    else:
        boundaries = [0]
        for left_peak, right_peak in zip(peaks_indices[:-1], peaks_indices[1:]):
            local_min = left_peak + int(np.argmin(p[left_peak:right_peak + 1]))
            boundaries.append(local_min + 1)
        boundaries.append(N)
        segments = list(zip(boundaries[:-1], boundaries[1:]))

    peaks_data = {}
    peaks_mass = 0

    for j, (a, b) in enumerate(segments, start=1):
        if periodic and a >= b:
            indices = np.concatenate([np.arange(a, N), np.arange(0, b)])
            if a == b:
                indices = np.arange(0, N)
        else:
            indices = np.arange(a, b)

        local_mass = masses[indices]
        m = float(np.sum(local_mass))

        local_x = grid_centers[indices]

        if periodic:
            theta = 2 * np.pi * (local_x - left) / L

            S = np.dot(np.sin(theta), local_mass) / m
            C = np.dot(np.cos(theta), local_mass) / m

            center_theta = np.arctan2(S, C)
            center = float(left + (center_theta % (2 * np.pi)) * L / (2 * np.pi))

            dx = local_x - center
            dx = dx - L * np.round(dx / L)
            variance = float(np.dot(dx ** 2, local_mass) / m)

            boundary_rms = np.nan

        else:
            center = float(np.dot(local_x, local_mass) / m)
            variance = float(np.dot((local_x - center) ** 2, local_mass) / m)

            distance_to_left = center - left
            distance_to_right = right - center

            if distance_to_left < max_peak_std:
                boundary_rms = np.sqrt(float(np.dot((local_x - left) ** 2, local_mass) / m))
            elif distance_to_right < max_peak_std:
                boundary_rms = np.sqrt(float(np.dot((right - local_x) ** 2, local_mass) / m))
            else:
                boundary_rms = np.nan

        std = np.sqrt(max(variance, 0.0))
        if std < max_peak_std * 0.5:
            peaks_mass += m

        if std < max_peak_std:
            peaks_data[int(j)] = {
                "mass": m,
                "center": center,
                "variance": variance,
                "std": std,
                "boundary_rms": boundary_rms
            }

    return peaks_mass, peaks_data

