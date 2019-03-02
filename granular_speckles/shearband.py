import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import sys


def find_jump(x_shift, myarray):
    array = np.hstack((np.ones(x_shift), myarray))
    jump = np.copy(myarray)
    for x in range(len(jump)):
        jump[x] = array[x + x_shift] - array[x]
    return np.argmin(jump) - int(x_shift / 2.)


def minimum_jump(myarray):
    """
    This function is used to obtain parameters for the shearband.

    Assumed the granular system has two regime, one fully decorrelated and
    one totally decorrelated, the shear band is the transition interval.
    It is not sharp.
    This function measures the transition on all plausible lengths:
        1. The increment between two abscissa points (height in meta-pixel)
            is measured.
        2. All the possible combination of abscissa are measured, this is done
            increasing the distance between the points.
        3. For each distance the minimum value, and its argument are stored.
    The MinimumJump hypothesis is the following:
        the shortest distance that scores the minimum (maximum) value is the
        correct transition length.

    Parameters:
    ==========
    array: the 2D array which describes the smooth transition

    Returns:
    ========
    jumps: 3D array with:
            1. Jump length
            2. Max value of jump
            3. ArgMax of jump
    """
    minima = []
    for x in range(0, len(myarray)):
        jumps = find_jump(x, myarray)
        minima.append([x, np.min(jumps), np.argmin(jumps)])
    jumps = np.array(minima)
    height = np.abs(np.min(jumps[:, 1]) - np.max(jumps[:, 1]))
    low = np.max(jumps[:, 1])
    jumps[:, 1] = -(jumps[:, 1] - low) / height
    return jumps
    # print (jumps[-1])


def shearband_width(data, plot=False):
    """
    Measure the shearband width with the minimum jump hypothesis.

    The shear band is detected and its length is fitted by an exponential
    function.
    """

    def best_jump_exponential(my_jumps):
        def exp(x, m):
            return 1 - np.exp(-m * x)

        popt, pcov = curve_fit(exp, my_jumps[:, 0], my_jumps[:, 1], bounds=(0.001, 4))
        m = popt[0]
        return 1. / m

    def best_jump_first_maxima(my_jumps):
        for arg, value in zip(my_jumps[:, 0], my_jumps[:, 1]):
            if value > 0.65:
                return 10
        return int(arg)

    mode = "first_maxima"
    mode = "exponential"
    matrices = data["matrices"]
    speeds = data["speeds"]
    colors = data["colors"]
    jump_lengths = [minimum_jump(matrix) for matrix in matrices]
    first_maxima = [best_jump_first_maxima(seq) for seq in jump_lengths]
    exponential = [best_jump_exponential(seq) for seq in jump_lengths]
    if plot:
        fig5, ax8 = plt.subplots(1, 1)
        for enum, color, seq, speed in zip(range(len(speeds)), colors, jump_lengths, speeds):
            ax8.plot(seq[:, 0], seq[:, 1], label=speed, color=color)

            if mode is "first_maxima":
                ax8.scatter(first_maxima[enum], seq[int(first_maxima[enum]), 1], color=color)
                ax8.plot([seq[1, 0], seq[-1, 0]], [seq[int(first_maxima[enum]), 1],
                                                   seq[int(first_maxima[enum]), 1]], color=color)

            if mode is "exponential":
                arg = exponential[enum]
                ax8.plot(seq[:, 0], [(1 - np.exp(-1. / arg * x)) for x in seq[:, 0]], color=color)
                ax8.scatter(arg, 1 - np.exp(-1), color=color)
        ax8.legend()
    return first_maxima, jump_lengths


def shearband_from_correlation(data, axes=None):
    """
    This function extrapolate the trend of shear band given a 2D matrix with
    correlation time-lengths.
    """
    if axes is not None:
        plot = True
        axes2, ax3 = axes
    else:
        axes2 = [""]*len(data["colors"])

    def linear(x, m, b):
        return m * x + b

    matrices = data["matrices"]
    speeds = data["speeds"]
    colors = data["colors"]
    # print(len(best_jump), len(colors), len(scaled_matrices), len(speeds))
    matrix_jumps = []
    sigma_jump = []
    mid_points = []
    for ax2, c, mat, speed in zip(axes2, colors, matrices, speeds):
        jumps = np.array([find_jump(jump, mat) for jump in range(7,18)])
        mid_point = int(np.mean(jumps))
        sigma = abs(mid_point - np.percentile(jumps,100))

        jumps = [mid for mid in jumps if abs(mid_point - mid) < sigma + 3]
        sigma = abs(mid_point - np.percentile(jumps,100))
        mid_point = int(np.mean(jumps))
        # mid_point = find_jump(10,mat)
        start_point = int(mid_point + int(sigma))
        end_point = int(mid_point - int(sigma))
        # print(length, mid_point)
        sigma_jump.append(sigma)
        matrix_jumps.append(mid_point)
        mid_points.append(jumps)
        if axes:
            # ax1.plot(jumps / np.min(jumps), ls="--", alpha=0.5, color=c)
            ax2.plot(mat, label=str(speed)+" m/s", color=c)
            ax2.scatter(mid_point, mat[mid_point], color=c)
            ax2.scatter(end_point, mat[end_point], color=c, marker=".")
            ax2.scatter(start_point, mat[start_point], color=c, marker=".")
            ax2.plot([start_point, end_point],
                     [mat[mid_point], mat[mid_point]], ls="--", color=c)
            ax2.bar(mid_point,1,abs(start_point-end_point),color=c, alpha=0.60)
            ax3.scatter(speed, mid_point, color=c)
            ax3.errorbar(speed, mid_point, yerr=sigma, color=c)
            ax3.semilogx()
    popt, pcov = curve_fit(linear, speeds, matrix_jumps)
    if axes:
        ax3.plot(speeds, speeds * popt[0] + popt[1], color="red")
    # return np.array([popt[0], popt[1]])##, pcov[0], pcov[1]]
    return matrix_jumps, sigma_jump, mid_points


def robustezza(matrices, mobmean_range, x_shift_range):
    y0 = x_shift_range[0]
    x0 = mobmean_range[0]
    robust = np.ndarray((mobmean_range[1] - x0,
                         x_shift_range[1] - y0, 2))
    for x in range(mobmean_range[0], mobmean_range[1]):
        for y in range(x_shift_range[0], x_shift_range[1]):
            robust[x - x0, y - y0] = shearband_from_correlation(data,
                                                                mobmean=x, x_shift=y)
    return robust


def shearband(speeds, matrices, plot=False):
    # type: (np.array, np.array, np.array) -> object
    # average_matrices = np.mean(matrices, axis=2)
    # clean_matrices = [list(mobilmean(smooth_mobilmean,x)) for x in average_matrices]
    # matrices = [x/np.max(x) for x in clean_matrices]
    colors = cm.rainbow(np.linspace(0, 1, len(speeds)))
    data = {"speeds": speeds, "matrices": matrices, "colors": colors}
    # best_jump, jump_params = shearband_width(data, plot)
    # data["best_jump"] = best_jump
    # data["jump_params"] = jump_params
    matrix_jumps, sigma_jump, mid_points = shearband_from_correlation(data)
    plot = True
    if plot:
        fig, ax3 = plt.subplots(1, 1)
        fig2, axes2 = plt.subplots(len(colors),1, sharex=True)
        axes2[0].set_title("Shear band detection")
        axes2[-1].set_xlabel("Distance from top of the well (metapixel)")
        # ax2.set_xlabel("Distance from bottom (pixel)")
        # ax2.set_ylabel("Scaled correlation time (frame/rate)")
        ax3.set_title("Share band over plate-rotation speed")
        ax3.set_xlabel("Rotating plate speed (m/s)")
        ax3.set_ylabel("Shear band depth (metapixels)")
        shearband_from_correlation(data, axes=(axes2, ax3))
        fig.tight_layout()
        fig2.tight_layout()
        fig2.legend()
        # fig2.show()
        # fig.show()

    return np.array([matrix_jumps, sigma_jump, speeds]), mid_points

# if False:
#     fig3, ax5 = plt.subplots(1, 1)
#     robust=robustezza(matrices,[2,25],[2,25])
#     ax5.set_title("Robustness for shear band slope changing analysis parameters")
#     ax5.set_xlabel("window for smooth filtering")
#     ax5.set_ylabel("Incremental ratio shift")
#     ax5.pcolor(robust[:,:,0])
#
#     fig4, ax6 = plt.subplots(1, 1)
#     ax6.set_title("Robustness for shear band intercept changing analysis parameters")
#     ax6.set_xlabel("window for smooth filtering")
#     ax6.set_ylabel("Incremental ratio shift")
#     ax6.pcolor(robust[:,:,1])
#
