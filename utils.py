import os
from typing import Tuple, List, Union, Dict
from datetime import datetime
import random
import logging
import numpy as np
from numpy import ndarray as array
import torch
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from scipy.stats import entropy

def get_eigenvalues(factors: List[Tensor],
                    verbose: bool = False) -> Tensor:
    """Computes the eigenvalues of KFAC, EFB or diagonal factors.

    Args:
        factors: A list of KFAC, EFB or diagonal factors.
        verbose: Prints out progress if True.

    Returns:
        The eigenvalues of all KFAC, EFB or diagonal factors.
    """
    eigenvalues = Tensor()
    factors = tqdm(factors, disable=not verbose)
    for layer, factor in enumerate(factors):
        factors.set_description(desc=f"Layer [{layer + 1}/{len(factors)}]")
        if len(factor) == 2:
            xxt_eigvals = torch.symeig(factor[0])[0]
            ggt_eigvals = torch.symeig(factor[1])[0]
            eigenvalues = torch.cat([eigenvalues, torch.ger(xxt_eigvals, ggt_eigvals).contiguous().view(-1)])
        else:
            eigenvalues = torch.cat([eigenvalues, factor.contiguous().view(-1)])
    return eigenvalues

def get_eigenvectors(factors: Dict[Module, Tensor]) -> Dict[Module, Tensor]:
    """Computes the eigenvectors of KFAC factors.

    Args:
        factors: A dict mapping layers to lists of first and second KFAC factors.

    Returns:
        A dict mapping layers to lists containing the first and second KFAC factors eigenvectors.
    """
    eigenvectors = dict()
    for layer, (xxt, ggt) in factors.items():
        sym_xxt, sym_ggt = xxt + xxt.t(), ggt + ggt.t()
        _, xxt_eigvecs = torch.symeig(sym_xxt, eigenvectors=True)
        _, ggt_eigvecs = torch.symeig(sym_ggt, eigenvectors=True)
        eigenvectors[layer] = (xxt_eigvecs, ggt_eigvecs)
    return eigenvectors

def kron(a: Tensor,
         b: Tensor) -> Tensor:
    r"""Computes the Kronecker product between the two 2D-matrices (tensors) `a` and `b`.

    `Wikipedia example <https://en.wikipedia.org/wiki/Kronecker_product>`_.

    Args:
        a: A 2D-matrix
        b: A 2D-matrix

    Returns:
        The Kronecker product between `a` and `b`.

    Examples:
        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> b = torch.tensor([[0, 5], [6, 7]])
        >>> kron(a, b)
        tensor([[ 0,  5,  0, 10],
                [ 6,  7, 12, 14],
                [ 0, 15,  0, 20],
                [18, 21, 24, 28]])
    """
    return torch.einsum("ab,cd->acbd", [a, b]).contiguous().view(a.size(0) * b.size(0), a.size(1) * b.size(1))

def negative_log_likelihood(probabilities: array,
                            labels: array) -> float:
    """Computes the Negative Log-Likelihood (NLL) of the predicted class probabilities.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.

    Returns:
        The NLL.
    """
    return -np.mean(np.log(probabilities[np.arange(probabilities.shape[0]), labels] + 1e-12))

def accuracy(probabilities: array,
             labels: array) -> float:
    """Computes the top 1 accuracy of the predicted class probabilities in percent.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.

    Returns:
        The top 1 accuracy in percent.
    """
    return 100.0 * np.mean(np.argmax(probabilities, axis=1) == labels)

def confidence(probabilities: array,
               mean: bool = True) -> Union[float, array]:
    """The confidence of a prediction is the maximum of the predicted class probabilities.

    Args:
        probabilities: The predicted class probabilities.
        mean: If True, returns the average confidence over all provided predictions.

    Returns:
        The confidence.
    """
    if mean:
        return np.mean(np.max(probabilities, axis=1))
    return np.max(probabilities, axis=1)

def calibration_curve(probabilities: array,
                      labels: array,
                      bins: int = 20) -> Tuple[float, array, array, array]:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.

    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` with equal number of samples.

    Source: `A Simple Baseline for Bayesian Neural Networks <https://arxiv.org/abs/1902.02476>`_.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.

    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    confidences = np.max(probabilities, 1)
    step = (confidences.shape[0] + bins - 1) // bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    # bins = np.linspace(0.1, 1.0, 30)
    predictions = np.argmax(probabilities, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = predictions == labels

    xs = []
    ys = []
    zs = []

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            xs.append(avg_confidence_in_bin)
            ys.append(accuracy_in_bin)
            zs.append(prop_in_bin)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    return ece, xs, ys, zs

def expected_calibration_error(probabilities: array,
                               labels: array,
                               bins: int = 10) -> Tuple[float, array, array, array]:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.

    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.

    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.

    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    conf = confidence(probabilities, mean=False)
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ece = 0
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = accuracy(probabilities[mask], labels[mask]) / 100
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            ece += mask.mean() * np.abs(ace)

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    return ece, np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)

def predictive_entropy(probabilities: array,
                       mean: bool = False) -> Union[array, float]:
    r"""Computes the predictive entropy of the predicted class probabilities.

    It is defined as :math:`H(y)=-\sum_{c=1}^K y_c\ln y_c` where `y_c` is the predicted class
    probability for class c and `K` is the number of classes.

    Args:
        probabilities: The predicted class probabilities.
        mean: If True, returns the average predictive entropy over all provided predictions.

    Returns:
        The predictive entropy.
    """
    pred_ent = np.apply_along_axis(entropy, axis=1, arr=probabilities)
    if mean:
        return np.mean(pred_ent)
    return pred_ent