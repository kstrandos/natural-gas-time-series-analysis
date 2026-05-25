
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

def utils_test(num1:int, num2:int):
    """Simple addition test to check if the utils import works"""
    return num1 + num2

def run_sequence_plot(x, y, title, xlabel="Time", ylabel="Values", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10, 3.5))

    ax.plot(x, y, 'k-')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    return ax
    
def mse(observations, estimates):

    # check arg types
    assert type(observations) == type(np.array([])), "'observations' must be a numpy array"
    assert type(estimates) == type(np.array([])), "'estimates' must be a numpy array"
    # check length of arrays equal
    assert len(observations) == len(estimates), "Arrays must be of equal length"

    # calculations
    difference = observations - estimates
    sq_diff = difference ** 2
    mse = np.mean(sq_diff)

    return mse

def fft_analysis(signal):

    # Linear detrending
    slope, intercept = np.polyfit(np.arange(len(signal)), signal, 1)
    trend = np.arange(len(signal)) * slope + intercept
    detrended = signal - trend

    fft_values = fft(detrended)
    frequencies = np.fft.fftfreq(len(fft_values))

    # Remove negative frequencies and sort
    positive_frequencies = frequencies[frequencies > 0]
    magnitudes = np.abs(fft_values)[frequencies > 0]

    # Identify dominant frequency
    dominant_frequency = positive_frequencies[np.argmax(magnitudes)]
    print(f"Dominant Frequency: {dominant_frequency:.3f}")

    # Convert frequency to period (e.g., days, weeks, months, etc.)
    dominant_period = 1 / dominant_frequency
    print(f"Dominant Period: {dominant_period:.2f} time units")

    return dominant_period, positive_frequencies, magnitudes

# Utility function to make the plots
def seas_decomp_plots(original, decomposition):
    _, axes = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(7, 5))
    axes[0].plot(original, label='Original')
    axes[0].legend(loc='upper left')
    axes[1].plot(decomposition.trend, label='Trend')
    axes[1].legend(loc='upper left')
    axes[2].plot(decomposition.seasonal, label='Seasonality')
    axes[2].legend(loc='upper left')
    axes[3].plot(decomposition.resid, label='Residuals')
    axes[3].legend(loc='upper left')
    plt.show()
    
def moving_average(observations, window=3, forecast=False):
    cumulative_sum = np.cumsum(observations, dtype=float)
    cumulative_sum[window:] = cumulative_sum[window:] - cumulative_sum[:-window]
    ma = cumulative_sum[window - 1:] / window
    if forecast:
        observations = np.append(observations, np.nan)
        ma_forecast = np.insert(ma, 0, np.nan*np.ones(window))
        return observations, ma_forecast
    else:
        return ma
    
    
def weighted_moving_average(observations, weights, forecast=False):

    if len(weights) != len(observations[0:len(weights)]):
        raise ValueError("Length of weights must match the window size")

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Initialize the result array
    result = np.empty(len(observations)-len(weights)//2)
    result[:] = np.nan

    # Calculate weighted moving average
    for i in range(len(weights)//2, len(result)):
        window = observations[i-(len(weights)//2):i+len(weights)//2+1]
        result[i] = np.dot(window, weights)

    # Handle forecast option
    if forecast:
        result = np.insert(result, 0, np.nan*np.ones(len(weights)//2+1))
        observations = np.append(observations, np.nan*np.ones(len(weights)//2))
        return observations, result
    else:
        return result