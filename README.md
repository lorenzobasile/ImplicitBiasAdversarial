# Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension

Code for the paper.

Required libraries, data and models can be downloaded by running:

```makefile
bash preparation.sh
```

Then, the first step is to obtain trained models, which can by done by typing:

```
python train.py --model
```

The `model` argument can be chosen between `resnet20` and `cct` (for CIFAR-10) and `resnet18` (for Imagenette). This script will create a `trained_models` folder containing the trained parameters for the chosen model, to be used for the following experiments.

Fourier masks (essential frequency masks and adversarial frequency masks) can be trained by running:

```
python mask_train.py --model --attack --mask
```

The first argument follows the same syntax as above, while `attack` determines the kind of adversarial attack to employ and it can be chosen among `FMN`, `PGD` and `DF` (for respectively Fast Minimum Norm , Projected Gradient Descent  and DeepFool). `mask` defines the type of masks to be trained, either `essential` or `adversarial`.

Finally, once both essential frequency masks and adversarial frequency masks have been computed for a given model-attack pair, correlations (based on cosine similarity and on the novel method based on Intrinsic Dimension) can be computed using:

```
python correlation.py --model --attack
```

