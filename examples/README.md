# Examples

Example python notebooks for pyDeepInsight.

## [pyDeepInsight - pyTorch - SqueezeNet](./pytorch_squeezenet.ipynb)

Classification of [TCGA][tcga] cancer type using RNA-seq data (see [data](./data/) directory) via **pyDeepInsight**. It 
provides code to convert tabular data to image format using the `ImageTransformer` class and t-SNE. The transformed 
is used to train a [torchvision SqueezeNet][tvsn] model.

## [CAM-based Feature Selection](./cam_feature_selection.ipynb)

An abridged version of the TCGA cancer type classification using **pyDeepInsight**, this notebook applies 
the `ImageTransformer` class with UMAP and a [timm][timm] ResNet50 model. It also demonstrates the use of 
the `CAMFeatureSelector` class for identifying potentially important features from the trained model.

## [MRep-DeepInsight Madelon Dataset](./mrep_madelon.ipynb)

A reproduction of the [Madelon dataset][made] analysis using **MRep-DeepInsight** as presented 
in [Sharma et al., 2024][mrep]. This example introduces the `MRepDeepInsight` class, demonstrates two 
data augmentation techniques, and uses PyTorch [Lightning][ligh] and [Optuna][opta] for model training and 
hyperparameter optimization.

## [MRepDeepInsight Madelon Dataset (tensor)](./mrep_madelon_tensor.ipynb)

A variant of the previous notebook, this version transforms the input directly into PyTorch-formatted 
tensors (`N, C, H, W`) rather than using NumPy/PIL-style arrays (`N, H, W, C`). This approach is preferred when 
no additional external image preprocessing is required, offering a more streamlined pipeline.

## [Discretization Methods](./discretization_methods.ipynb) 

This notebook compares the five primary discretization methods available in **pyDeepInsight**: `bin`, `qtb`, `lsa`, 
`sla`, and `ags`. It provides a visual demonstration of how each method maps features to pixels, along with a runtime 
performance comparison.

[tcga]: https://www.cancer.gov/ccg/research/genome-sequencing/tcga
[tvsn]: https://pytorch.org/vision/main/models/squeezenet.html
[timm]: https://timm.fast.ai/
[made]: https://archive.ics.uci.edu/dataset/171/madelon
[mrep]: https://doi.org/10.1038/s41598-024-63630-7
[ligh]: https://lightning.ai/docs/pytorch/stable/
[opta]: https://optuna.org/



