import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from scipy.signal import find_peaks
from scipy import stats
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
path = 'C:\\Users\\mikov\\Documents\\Programmeren\\Year 2\\SDA\\HW\\Final Project\\'

def find_kde_valley(logR_sample, bw_method=None):
    """
    Find the radius valley between the two peaks in the PDF
    of log(radius) using a Gaussian KDE. The function selects
    the valley between the two most prominent peaks.
    
    The result is sensitive to ``bw_method``. You are encouraged
    to experiment with both the KDE bandwidth and the parameters
    of ``find_peaks``.

    Parameters
    ----------
    logR_sample : np.ndarray
        Array with log(radius) samples.
    bw_method : float
        Bandwidth for the KDE.

    Returns
    -------
    float, None
        The radius valley, as log(R_valley). Returns None if fewer
        than two peaks are found or no valid valley exists.
    np.ndarray
        The grid of log(R_valley) values over which the KDE was evaluated.
    np.ndarray
        The KDE for the grid of log(R_valley) values.
    """

    kde = stats.gaussian_kde(logR_sample, bw_method=bw_method)

    logr_grid = np.linspace(np.min(logR_sample), np.max(logR_sample), 200)
    pdf = kde(logr_grid)

    peaks, props = find_peaks(pdf, prominence=0.01*np.max(pdf))
    valleys, _ = find_peaks(-pdf)

    log_radius_valley = None
    
    if len(peaks) >= 2:
        top_two = peaks[np.argsort(pdf[peaks])[-2:]]
        top_two = np.sort(top_two)

        valley_candidates = valleys[
            (logr_grid[valleys] > logr_grid[top_two[0]]) &
            (logr_grid[valleys] < logr_grid[top_two[1]])
        ]

        if len(valley_candidates) > 0:
            valley_idx = valley_candidates[np.argmin(pdf[valley_candidates])]
            log_radius_valley = logr_grid[valley_idx], 

    return log_radius_valley, logr_grid, pdf

def log_likelihood(a, b, logP, logR_valley, sigma):
    '''
    Function to compute the log-likelihood of the linear model

    parameters:
    a : float
        Intercept of the linear model.
    b : float
        Slope of the linear model.
    logP : np.ndarray
        Array of log(orbital period) values for each bin.
    logR_valley : np.ndarray
        Array of log(radius valley) values for each bin.
    sigma : np.ndarray
        Array of uncertainties for log(radius valley) values for each bin.

    returns:
    float
        The log-likelihood of the linear model given the data.
    '''
    model = a + b * logP
    residuals = logR_valley - model
    return -0.5 * np.sum((residuals / sigma) ** 2)

# 1. Data extraction
data = ascii.read(path + "exoplanet_data.csv") # imports the dataset

# 2. Data filtering
print(f'Before filtering: {len(data)} planets')
mask = (data['P'] < 50) & (data['P'] > 1) & (data['Rplanet'] < 4) & (data['Rplanet'] > 1) & (data['Rplanet_sigma'] / data['Rplanet'] < 0.2)
data_filtered = data[mask]
print(f'After filtering: {len(data_filtered)} planets')

# 3. Creating bins
period_bins = np.logspace(0, 1, 5) # creates 4 logarithmically spaced bins between 1 and 10

# 4. Scatter plot
plt.loglog()
plt.scatter(data_filtered['P'], data_filtered['Rplanet'], s=10, alpha=0.5, label='Planet data')

x_fit = np.linspace(np.min(data_filtered['P']), np.max(data_filtered['P']), 1000)
a, b = 0.4, -0.2
y_fit = 10**a * x_fit**b
plt.plot(x_fit, y_fit, color='red', label='Linear fit: ' + r'$\log R_\text{valley} = $' + f'{a:.2f} + {b:.2f}' + r'$\log P$')

plt.xlabel('Orbital Period (days)')
plt.ylabel('Planet Radius (Earth radii)')
plt.title('Exoplanet Radius vs. Orbital Period')
plt.legend()
plt.savefig(path + 'Exoplanet Radius vs Orbital Period.pdf')
plt.show()

# 5. R distribution and KDE
logR_sample = np.log10(data_filtered['Rplanet'])
plt.hist(logR_sample, bins=30, density=True, alpha=0.5, label='Histogram of log(R)', edgecolor='black')
for bw in [0.1, 0.2, 0.4]:
    logR_valley, logr_grid, pdf = find_kde_valley(logR_sample, bw_method=bw)
    if logR_valley is not None:
        print(f'for bandwidth {bw}, log(R_valley) = {10**logR_valley[0]:.2f} Earth radii')
    plt.plot(logr_grid, pdf, label=f'KDE (bw={bw})')
plt.xlabel('log(R) (log Earth radii)')
plt.ylabel('Density')
plt.title('Distribution of Planet Radii')
plt.legend()
plt.savefig(path + 'Distribution of Planet Radii.pdf')
plt.show()

# 6. Bootstrap resampling per period bin
logR, logP, sigma = [], [], []

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
subplot_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

for bin_index in range(len(period_bins) - 1):
    p_bounds = period_bins[bin_index: bin_index + 2]

    bin_mask = (data_filtered['P'] >= p_bounds[0]) & (data_filtered['P'] < p_bounds[1])
    bin_data = data_filtered[bin_mask]

    print(f'Bin {bin_index + 1}: {p_bounds[0]:.2f} - {p_bounds[1]:.2f} days, {len(bin_data)} planets')

    bin_bootstrap_valleys = []

    for i in range(1000):
        sample_indices = np.random.randint(0, len(bin_data), len(bin_data))
        sampled_radii = np.asarray(bin_data['Rplanet'])[sample_indices]
        sampled_sigma = np.asarray(bin_data['Rplanet_sigma'])[sample_indices]

        rng_radii = np.random.normal(loc=sampled_radii, scale=sampled_sigma)
        rng_radii = rng_radii[rng_radii > 0]

        logR_bootstrap = np.log10(rng_radii)
        logR_valley, grid, KDE_vals = find_kde_valley(logR_bootstrap, bw_method=0.2)

        if logR_valley is not None:
            bin_bootstrap_valleys.append(logR_valley[0])

    bin_bootstrap_valleys = np.asarray(bin_bootstrap_valleys)

    median = np.median(bin_bootstrap_valleys)
    p16, p84 = np.percentile(bin_bootstrap_valleys, [16, 84])

    logP.append(np.log10(np.mean(p_bounds)))
    logR.append(median)
    sigma.append((p84 - p16) / 2)

    ax_idx = subplot_indices[bin_index]
    ax[ax_idx].hist(bin_bootstrap_valleys, bins=30, density=True, alpha=0.5, edgecolor='black')
    ax[ax_idx].axvline(p16, color='red', linestyle='--', label=f'16th percentile: {10**p16:.2f} $R_\oplus$')
    ax[ax_idx].axvline(median, color='blue', linestyle='-', label=f'Median: {10**median:.2f} $R_\oplus$')
    ax[ax_idx].axvline(p84, color='red', linestyle='--', label=f'84th percentile: {10**p84:.2f} $R_\oplus$')
    ax[ax_idx].set_xlabel('log(R_valley) (log Earth radii)')
    ax[ax_idx].set_ylabel('Density')
    ax[ax_idx].set_title(f'bin {bin_index + 1}, {p_bounds[0]:.2f} - {p_bounds[1]:.2f} days')
    ax[ax_idx].legend()
    ax[ax_idx].set_xlim(0, 0.6)
    ax[ax_idx].grid()

fig.suptitle('Bootstrap Distributions of log(R_valley) for Each Period Bin')
plt.tight_layout()
plt.savefig('Bootstrap Distributions of log(R_valley) for Each Period Bin subplot.pdf')
plt.show()

logP = np.array(logP)
logR = np.array(logR)
sigma = np.array(sigma)

a_vals, b_vals = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)

L = np.empty((len(a_vals), len(b_vals)))

for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        L[i, j] = log_likelihood(a, b, logP=logP, logR_valley=logR, sigma=sigma)

# Plot heatmap
plt.figure(figsize=(9,7.5))
plt.imshow(L, origin='lower', extent=[b_vals[0], b_vals[-1], a_vals[0], a_vals[-1]], aspect='auto', cmap='magma')
plt.xlabel('b')
plt.ylabel('a')
plt.title('Log-likelihood over (a, b)')
plt.colorbar(label='Log-likelihood')

# Mark maximum-likelihood location
imax = np.unravel_index(np.argmax(L), L.shape)
best_a, best_b = a_vals[imax[0]], b_vals[imax[1]]
plt.scatter(best_b, best_a, color='blue', marker='x', label=f'Max Likelihood: a={best_a:.3f}, b={best_b:.3f}', zorder=5)

sigma_vals = np.array([1.0, 3.0, 5.0])
target_probs = 1.0 - np.exp(-0.5 * sigma_vals ** 2) 

L_flat = L.ravel()
post_unnorm = np.exp(L_flat - np.max(L_flat))
post_norm = post_unnorm / np.sum(post_unnorm)

sort_idx = np.argsort(-post_norm)
sorted_post = post_norm[sort_idx]
sorted_L = L_flat[sort_idx]
cumsum = np.cumsum(sorted_post)

levels = []
for p in target_probs:
    idx = np.searchsorted(cumsum, p)
    if idx >= len(sorted_L):
        lvl = sorted_L[-1]
    else:
        lvl = sorted_L[idx]
    levels.append(lvl)

# Ensure levels are in ascending order for contour and create matching labels
levels = np.array(levels)
labels = np.array([f'{int(sigma_val)}' + r'$\sigma$' for sigma_val in sigma_vals])
order = np.argsort(levels)
levels_sorted = levels[order]
labels_sorted = labels[order]

colors_contour = ['red', 'cyan', 'green']
B, A = np.meshgrid(b_vals, a_vals)
cs = plt.contour(B, A, L, levels=levels_sorted, colors=colors_contour, linewidths=1.2)

for i in range(len(levels_sorted)):
    plt.plot([], [], color=colors_contour[i], linewidth=1.2, label=labels_sorted[i])

plt.legend(loc='lower left')
plt.savefig(path + 'Log-likelihood over (a, b).pdf')
plt.show()

L_b0 = [log_likelihood(a=a_temp, b=0, logP=logP, logR_valley=logR, sigma=sigma) for a_temp in a_vals]
max_likelihood_b0 = np.max(L_b0)
print(f'Max likelihood at b=0: {max_likelihood_b0:.3f} with a={a_vals[np.argmax(L_b0)]:.3f}')

max_likelihood = log_likelihood(a=best_a, b=best_b, logP=logP, logR_valley=logR, sigma=sigma)
print(f'Max likelihood at best fit: {max_likelihood:.3f}')
likelihood_ratio = np.exp(max_likelihood_b0 - max_likelihood)
print(f'Likelihood ratio (b=0 vs best fit): {likelihood_ratio:.3e}')


print('The likelihood ratio is much smaller than 1, close to 0. Meaning that the likelihood favors a model with a non zero slope.')