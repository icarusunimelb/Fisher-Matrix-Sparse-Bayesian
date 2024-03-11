# [Sparse Bayesian Learning based on Last Layer Laplace Approximation and Low Rank Hessian Approxomation](https://github.com/icarusunimelb/Fisher-Matrix-Sparse-Bayesian)
This repository contains the source code for employing Laplace Approximation to estimtate the model uncertainty in deep neural networks. To reduce the space complexity and computational burden in inferencing the covariance matrix, last layer inference, kronecker-factored approximate curvature and low rank approximation are employed. 

## Dependencies
This repository leans heavily on the following Python packages: python3, numpy, pandas, scikit-learn, matplotlib, math, pytorch, torchvision, tqdm, seaborn, colorcet, statsmodels.

## References for Code Base
Curvatures: [https://github.com/DLR-RM/curvature](https://github.com/DLR-RM/curvature).
Langevin Dynamics: [https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics](https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics).
ResNet: [https://github.com/bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification).
Adversarial attack: [https://pytorch.org/tutorials/beginner/fgsm_tutorial.html](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html).
Toy classification problem: [https://github.com/wiseodd/last_layer_laplace](https://github.com/wiseodd/last_layer_laplace).
