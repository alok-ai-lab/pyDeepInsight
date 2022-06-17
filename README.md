# pyDeepInsight

This package provides a python implementation of 
[alok-ai-lab/DeepInsight](https://github.com/alok-ai-lab/DeepInsight) as originally 
described in [*DeepInsight: A methodology to transform a non-image data to an 
image for convolution neural network architecture*][1] [[1]](#1). This is not guaranteed to 
give the same results as the published MatLab code and should be considered 
experimental.

[1]: https://doi.org/10.1038/s41598-019-47765-6

## Installation
    python3 -m pip -q install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight

## Overview

The pyDeepInsight package provides classes that aid in the transformation of 
non-image data into image matrices that can be used to train a CNN. 

<a id='imagetransformer'></a>
### ImageTransformer Class

Transforms features to an image matrix using dimensionality reduction and 
discritization.

```python
class pyDeepInsight.ImageTransformer(feature_extractor='tsne', 
discretization='bin', pixels=(224, 224))
```

#### Parameters

* **feature_extractor**: string of value ('*tsne*', '*pca*', '*kpca*') or a class 
instance with method 'fit_transform' that returns a 2-dimensional 
array of extracted features. The string values use the same parameters as those 
described in the [original paper][1]. Providing a class instance is preferred and 
allows for greater customization of the feature extractor, such as modifying 
perplexity in t-SNE, or use of alternative feature extractors, such as [UMAP][2].
* **discretization**: string of values ('*bin*', '*assignment*'). Defines the 
method for discretizing dimensionally reduced data to pixel coordinates.  
The default '*bin*' is the method implemented in the [original paper][1] and 
maps features to pixels based on a direct scaling of the extracted features to 
the pixel space.  
The '*assignment*' values applies SciPy's [solution to the linear sum 
assignment problem][3] to the exponent of the euclidean distance between the 
extracted features and pixels to assign a features to pixels with no overlap. 
In cases where the number of features exceeds the number of pixels, the 
features are clustered using K-means clustering, with *k* equal to the number 
of pixels, and those clusters are assigned to pixels.
* **pixels**: int (square matrix) or tuple of ints (height, width) that defines 
the size of the image matrix. A default of 224 × 224 is used as that is the 
minimum size expected by [torchvision models][4].

[2]: https://umap-learn.readthedocs.io/en/latest/
[3]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
[4]: https://pytorch.org/vision/stable/models.html

<a id='camfeatureselector'></a>
### CAMFeatureSelector Class

Extracts important features from a trained PyTorch model using class activation mapping
(CAM) as proposed in [*DeepFeature: feature selection in nonimage data using 
convolutional neural network*][5] [[2]](#2).

[5]: https://doi.org/10.1093/bib/bbab297
[6]: https://pytorch.org/

```python
class DeepInsight.CAMFeatureSelector(model, it, target_layer, cam_method="GradCAM")
```

#### Parameters
* **model**: a [pytorch.nn.Module][7] CNN model trained on the output from an
ImageTransformer instance. The [torchvision.models][4] subpackage provides many 
common CNN architechtures. 
* **it**: the [ImageTransformer](#imagetransformer) instance used to generate
the images used to train the **model**.
* **target_layer**: the target layer of the **model** on which the CAM is computed.
Can be specified using the name provided by [nn.Module.named_modules][8] or a 
by providing a pointer to the layer directly. If no layer is specified, the 
last non-reduced convolutional layer is selected as determined by 
the 'locate_candidate_layer' method of the [TorchCAM][9] [[3]](#3) package by 
François-Guillaume Fernandez.
* **cam_method**: the name of a CAM method class provided by the 
[pytorch_grad_cam][10] [[4]](#4) package by Jacob Gildenblat. Default is "GradCAM".

[7]: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
[8]: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_modules
[9]: https://github.com/frgfm/torch-cam
[10]: https://github.com/jacobgil/pytorch-grad-cam

<a id='logscaler'></a>
### LogScaler Class

Performs log normalization and scaling procedure as described as norm-2 in the
[DeepInsight paper supplementary information][13].
This method attempts keep the topology of the features by using a global maximum 
in the logarithmic scale. 

[13]: https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-019-47765-6/MediaObjects/41598_2019_47765_MOESM1_ESM.pdf

```python
>>> from pyDeepInsight import LogScaler
>>> import numpy as np
>>> data = np.ndarray([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
>>> scaler = LogScaler()
>>> _ = scalar.fit(data)
>>> print(scaler._min0, scaler._max)
[-1.  2.] 3.044522437723423
>>> print(scaler.transform(data))
[[0.         0.52863395]
 [0.13317856 0.72169761]
 [0.22767025 0.84248003]
 [0.36084881 1.        ]]
>>> print(scaler.transform([[2, 2]]))
[0.4553405  0.52863395]
```

## Example Jupyter Notebooks

* [Classification of TCGA data using SqueezeNet](./examples/pytorch_squeezenet.ipynb)
* [Feature Selection using GradCAM](./examples/pytorch_squeezenet.ipynb)

## References

<a id="1">[1]</a>
Sharma A, Vans E, Shigemizu D, Boroevich KA, & Tsunoda T. DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture. *Sci Rep* **9**, 11399 (2019). https://doi.org/10.1038/s41598-019-47765-6

<a id="2">[2]</a>
Sharma A, Lysenko A, Boroevich KA, Vans E, & Tsunoda T. DeepFeature: feature selection in nonimage data using convolutional neural network, *Briefings in Bioinformatics*, Volume 22, Issue 6, November 2021, bbab297, https://doi.org/10.1093/bib/bbab297

<a id="3">[3]</a>
François-Guillaume Fernandez. (2020). TorchCAM: class activation explorer. https://github.com/frgfm/torch-cam

<a id="4">[4]</a>
Jacob Gildenblat, & contributors. (2021). PyTorch library for CAM methods. https://github.com/jacobgil/pytorch-grad-cam
