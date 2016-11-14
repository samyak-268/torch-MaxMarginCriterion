# MaxMargin Criterion

This is a Torch implementation of the margin maximizing loss function that has been used in [1][1] and [2][2].

```lua
criterion = nn.MaxMarginCriterion([bias], [margin])
```

The loss function can be used for training Siamese networks where the data is presented in the form of image pairs. It ensures that the distance between the embeddings of similar face pairs is less than `(bias - margin)` whereas those of dissimilar face pairs is greater than `(bias + margin)`. Mathematically, the loss function can be formulated as - 

![``` \mathbf{L}(\{ d_{ij}, y_{ij} \}) = \sum_{T} max(0, m - y_{ij}(b - d_{ij}^2))  ```](https://raw.githubusercontent.com/samyak-268/torch-LargeMarginCriterion/master/images/lossFunctionEqn.png)

`T` refers to the mini-batch of training samples where each sample is of the form `(d_{ij}, y_{ij})`. `d_{ij}` is the L2 distance between the embeddings of the image pair and y_{ij} is either +1 (similar face pair) or -1 (dissimilar face pair). If not specified the bias (`b`) and the margin (`m`) are set to default values of 1 and 0.1 respectively.

## References
[1] Hu et al., *Discriminative Deep Metric Learning for Face Verification in the Wild*, CVPR 2014

[2] Sharma et al., *Local Higher-Order Statistics (LHS) - Describing images with statistics of local non-binarized pixel patterns*, CVIU 2016

[1]: http://www.grvsharma.com/hpresources/sharma_lhs_pre.pdf
[2]: http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Hu_Discriminative_Deep_Metric_2014_CVPR_paper.pdf
