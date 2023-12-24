import colorcet as cc
import numpy as np
from matplotlib import pyplot as plt, colors, patheffects, offsetbox
from seaborn import distplot
from statsmodels.distributions.empirical_distribution import ECDF

from utils import (predictive_entropy, expected_calibration_error, confidence, accuracy, calibration_curve)

def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)

def regression_uncertainty(xs, ys, sgd_predictions, mean_predictions, std_predictions):
    plt.plot(xs.cpu().detach().numpy(), ys.cpu().detach().numpy(), marker="o", linestyle='None', color='black')
    plt.plot(xs.cpu().detach().numpy(), np.power(xs.cpu().detach().numpy(), 3), color='black') 
    plt.plot(xs.cpu().detach().numpy(), sgd_predictions.cpu().detach().numpy(), color='red') 
    plt.plot(xs.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy(), color='darkblue')
    plt.fill_between(xs.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()-std_predictions.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()+std_predictions.cpu().detach().numpy(), color='deepskyblue')
    plt.show()




