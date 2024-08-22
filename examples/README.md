# Examples

Example python notebooks for pyDeepInsight.

## pytorch_squeezenet.ipynb

Classification of [TCGA][tcga] cancer type using RNA-seq data (see [data](./data/) directory) via pyDeepInsight. It 
provides code to convert tabular data to image format using the ImageTransformer class and t-SNE as well 
as how to train a [torchvision SqueezeNet][tvsn] model using the transformed data.

## cam_feature_selection.ipynb

Abridged classification of the TCGA cancer type in the above example using the ImageTransformer class with 
UMAP and a [Pytorch Image Models (timm)][timm] ResNet50 model, followed by a detailed explaination on how to use the 
CAMFeatureSelector class to extract potentially important features.

## mrep_madelon.ipynb

A reproduction of the [Madelon dataset][made] analysis using multiple representation (MRep) DeepInsight as reported in 
 [Sharma A. et al. 2024][mrep]. This example introduces the use of the MRepDeepInsight class as well as two data 
augmentation methods and includes use of PyTorch [Lightning][ligh] and [Optuna][opta] for training and hyperparamter 
optimization.

[tcga]: https://www.cancer.gov/ccg/research/genome-sequencing/tcga
[tvsn]: https://pytorch.org/vision/main/models/squeezenet.html
[timm]: https://timm.fast.ai/
[made]: https://archive.ics.uci.edu/dataset/171/madelon
[mrep]: https://doi.org/10.1038/s41598-024-63630-7
[ligh]: https://lightning.ai/docs/pytorch/stable/
[opta]: https://optuna.org/



