from telnetlib import XASCII
import colorcet as cc
import numpy as np
import torch
from torch import Tensor
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

def toy_classification_plot(X, Y, X1_test, X2_test, Z, test_range):
    cmap = 'Blues'
    plt.figure(figsize=(6, 5))

    im = plt.contourf(X1_test, X2_test, Z, alpha=0.7, cmap=cmap, levels=np.arange(0.3, 1.01, 0.1))
    cbar = plt.colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Confidence', rotation=270)
    
    plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], c='coral', edgecolors='k', linewidths=0.5)
    plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], c='yellow', edgecolors='k', linewidths=0.5)
    plt.scatter(X[Y==2][:, 0], X[Y==2][:, 1], c='yellowgreen', edgecolors='k', linewidths=0.5)
    plt.scatter(X[Y==3][:, 0], X[Y==2][:, 1], c='violet', edgecolors='k', linewidths=0.5)
    plt.scatter(X[Y==4][:, 0], X[Y==4][:, 1], c='deeppink', edgecolors='k', linewidths=0.5)

    plt.xlim(test_range)
    plt.ylim(test_range)
    plt.xticks([])
    plt.yticks([])

    plt.xlabel("Feature 1",fontsize=16)
    plt.ylabel("Feature 2",fontsize=16)

    plt.show()

def toy_regression_plot(X: Tensor,
                        Y: Tensor, 
                        sgd_predictions: Tensor, 
                        mean_predictions: Tensor, 
                        std_predictions: Tensor):
    plt.plot(X.cpu().detach().numpy(), Y.cpu().detach().numpy(), marker="o", linestyle='None', color='black')
    plt.plot(X.cpu().detach().numpy(), np.power(X.cpu().detach().numpy(), 3), color='black') 
    plt.plot(X.cpu().detach().numpy(), sgd_predictions.cpu().detach().numpy(), color='red') 
    plt.plot(X.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy(), color='darkblue')
    plt.fill_between(X.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()-4*std_predictions.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()+4*std_predictions.cpu().detach().numpy(), color='lightblue')
    plt.fill_between(X.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()-3*std_predictions.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()+3*std_predictions.cpu().detach().numpy(), color='lightskyblue')
    plt.fill_between(X.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()-2*std_predictions.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()+2*std_predictions.cpu().detach().numpy(), color='skyblue')
    plt.fill_between(X.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()-std_predictions.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()+std_predictions.cpu().detach().numpy(), color='deepskyblue')
    plt.show()


def regression_uncertainty(xs, ys, sgd_predictions, mean_predictions, std_predictions):
    plt.plot(xs.cpu().detach().numpy(), ys.cpu().detach().numpy(), marker="o", linestyle='None', color='black')
    plt.plot(xs.cpu().detach().numpy(), np.power(xs.cpu().detach().numpy(), 3), color='black') 
    plt.plot(xs.cpu().detach().numpy(), sgd_predictions.cpu().detach().numpy(), color='red') 
    plt.plot(xs.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy(), color='darkblue')
    plt.fill_between(xs.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()-std_predictions.cpu().detach().numpy(), mean_predictions.cpu().detach().numpy()+std_predictions.cpu().detach().numpy(), color='deepskyblue')
    plt.show()




