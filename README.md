# Frequency maps reveal the correlation between Adversarial Attacks and Implicit Bias

Code for the paper, tested with `python>=3.8, <=3.10`.

Required libraries are reported in `requirements.txt` and they can be installed via `pip`. 

Data and models can be downloaded by running:

```makefile
bash preparation.sh
```

Then, the first step is to obtain trained models, which can by done by typing:

```
python train.py --model
```

The `model` argument can be chosen between `resnet20` (for CIFAR-10) and `resnet18` and `vit` (for Imagenette). This script will create a `trained_models` folder containing the trained parameters for the chosen model, to be used for the following experiments.

Fourier maps (essential frequency maps and adversarial frequency maps) can be trained by running:

```
python map_train.py --model --attack --map
```

The first argument follows the same syntax as above, while `attack` determines the kind of adversarial attack to employ and it can be chosen among `FMN`, `PGD` and `DF` (for respectively Fast Minimum Norm , Projected Gradient Descent  and DeepFool). `map` defines the type of maps to be trained, either `essential` or `adversarial`.

Once both essential frequency maps and adversarial frequency maps have been computed for a given model-attack pair, correlations (based on cosine similarity and on the novel method based on Intrinsic Dimension) can be computed using:

```
python correlation.py --model --attack
```

Finally, the class-level maps can be computed and tested by running:

```
python class_level_maps.py
```