# pyDeepInsight

This package provides a python implementation of 
[alok-ai-lab/DeepInsight](https://github.com/alok-ai-lab/DeepInsight) as originally 
described in [*DeepInsight: A methodology to transform a non-image data to an 
image for convolution neural network architecture*][di] [\[1\]](#1). This is not guaranteed to 
give the same results as the published MatLab code and should be considered 
experimental.



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
described in the [original paper][di]. Providing a class instance is preferred and 
allows for greater customization of the feature extractor, such as modifying 
perplexity in t-SNE, or use of alternative feature extractors, such as [UMAP][umap].
* **discretization**: string of values ('*bin*', '*assignment*'). Defines the 
method for discretizing dimensionally reduced data to pixel coordinates.  
The default '*bin*' is the method implemented in the [original paper][di] and 
maps features to pixels based on a direct scaling of the extracted features to 
the pixel space.  
The '*assignment*' values applies SciPy's [solution to the linear sum 
assignment problem][lsa] to the exponent of the euclidean distance between the 
extracted features and pixels to assign a features to pixels with no overlap. 
In cases where the number of features exceeds the number of pixels, the 
features are clustered using K-means clustering, with *k* equal to the number 
of pixels, and those clusters are assigned to pixels.
* **pixels**: int (square matrix) or tuple of ints (height, width) that defines 
the size of the image matrix. A default of 224 × 224 is used as that is the 
common minimum size expected by [torchvision][tv] and [timm][timm] pre-trained models.

#### Methods

* **fit**(X[, y=None, plot=False]): Compute the mapping of the feature space to the image space.
* **transform**(X[, y=None, img_format='rgb']): Perform feature space to image space mapping.
* **fit_transform**(X[, y=None]): Fit to data, then transform it.
* **pixel**([pixels]): Get or set the image dimensions 
* **inverse_transform**(img): Transform from the image space back to the feature space.

<a id='camfeatureselector'></a>
### CAMFeatureSelector Class

Extracts important features from a trained PyTorch model using class activation mapping
(CAM) as proposed in [*DeepFeature: feature selection in nonimage data using 
convolutional neural network*][df] [\[2\]](#2).

```python
class DeepInsight.CAMFeatureSelector(model, it, target_layer, cam_method="GradCAM")
```

#### Parameters
* **model**: a [pytorch.nn.Module][pytm] CNN model trained on the output from an
ImageTransformer instance. The [torchvision.models][tv] subpackage provides many 
common CNN architechtures. 
* **it**: the [ImageTransformer](#imagetransformer) instance used to generate
the images used to train the **model**.
* **target_layer**: the target layer of the **model** on which the CAM is computed.
Can be specified using the name provided by [nn.Module.named_modules][pytmname] or a 
by providing a pointer to the layer directly. If no layer is specified, the 
last non-reduced convolutional layer is selected as determined by 
the 'locate_candidate_layer' method of the [TorchCAM][tcam] [\[3\]](#3) package by 
François-Guillaume Fernandez.
* **cam_method**: the name of a CAM method class provided by the 
[pytorch_grad_cam][tgcam] [\[4\]](#4) package by Jacob Gildenblat. Default is "GradCAM".

#### Methods

* **calculate_class_activations**(X, y, [batch_size=1, flatten_method='mean']): Calculate 
CAM for each input then flatten for each class.
* **select_class_features**(cams, [threshold=0.6]): Select features for each class using 
class-specific CAMs. Input feature coordinates are filtered based on activation at same 
coordinates.

## Example Jupyter Notebooks

* [Classification of TCGA data using SqueezeNet](./examples/pytorch_squeezenet.ipynb)
* [Feature Selection using GradCAM](./examples/cam_feature_selection.ipynb)

## References

<a id="1">\[1\]</a>
Sharma A, Vans E, Shigemizu D, Boroevich KA, & Tsunoda T. DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture. *Sci Rep* **9**, 11399 (2019). https://doi.org/10.1038/s41598-019-47765-6

<a id="2">\[2\]</a>
Sharma A, Lysenko A, Boroevich KA, Vans E, & Tsunoda T. DeepFeature: feature selection in nonimage data using convolutional neural network, *Briefings in Bioinformatics*, Volume 22, Issue 6, November 2021, bbab297. https://doi.org/10.1093/bib/bbab297

<a id="3">\[3\]</a>
François-Guillaume Fernandez. (2020). TorchCAM: class activation explorer. https://github.com/frgfm/torch-cam

<a id="4">\[4\]</a>
Jacob Gildenblat, & contributors. (2021). PyTorch library for CAM methods. https://github.com/jacobgil/pytorch-grad-cam

[di]: https://doi.org/10.1038/s41598-019-47765-6
[umap]: https://umap-learn.readthedocs.io/en/latest/
[lsa]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
[tv]: https://pytorch.org/vision/stable/models.html
[timm]: https://github.com/rwightman/pytorch-image-models
[df]: https://doi.org/10.1093/bib/bbab297
[pyt]: https://pytorch.org/
[pytm]: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
[pytmname]: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_modules
[tcam]: https://github.com/frgfm/torch-cam
[tgcam]: https://github.com/jacobgil/pytorch-grad-cam
[disi]: https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-019-47765-6/MediaObjects/41598_2019_47765_MOESM1_ESM.pdf

# Citation
```
@ARTICLE{Sharma2019-rs,
  title     = "{DeepInsight}: A methodology to transform a non-image data to an
               image for convolution neural network architecture",
  author    = "Sharma, Alok and Vans, Edwin and Shigemizu, Daichi and
               Boroevich, Keith A and Tsunoda, Tatsuhiko",
  journal   = "Sci. Rep.",
  volume    =  9,
  number    =  1,
  pages     = "11399",
  year      =  2019,
}
```