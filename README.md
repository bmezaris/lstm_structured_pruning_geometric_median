## Structured Pruning of LSTMs via Eigenanalysis and Geometric Median

This repository contains the code for the paper “Structured Pruning of LSTMs via Eigenanalysis and Geometric Median for Mobile Multimedia and Deep Learning Applications” (Proc. 22nd IEEE Int. Symposium on Multimedia (ISM), Dec. 2020). The author's accepted version of the paper is available at https://www.iti.gr/~bmezaris/publications/ism2020_preprint.pdf. The final publication is available at https://ieeexplore.ieee.org/

## Introduction

In contrast to structured DCNN pruning, which has been extensively studied in the literature [1, 2, 3, 4], structured RNN pruning is a much less investigated topic [5, 6].
However, both [5, 6] utilize sparsity-inducing regularizers to modify the loss function, which may lead to numerical instabilities and suboptimal solutions [7].
Inspired from recent advances in DCNN filter pruning [2, 3] we extend [5] as following:
1) The covariance matrix formed by each layer’s responses is used to compute the respective eigenvalues, quantify the layer’s redundancy and pruning rate (as in [2] for DCNN layers).
2) A Geometric Median-based criterion is used to identify the most redundant LSTM units (as in [3] for DCNN filters).

## Dependencies

To run the code use TensorFlow 2.3 or later.

Download the YouTube-8M dataset (frame-level features) into the dbs folder.

## License and Citation

The code is provided for academic, non-commercial use only.
Please also check for any restrictions applied in the code parts used here from other sources (e.g. provided datasets, YouTube-8M Tensorflow Starter Code, etc.).
If you find the code useful in your work, please cite the following publication where this approach is described:

Bibtex:
```
@INPROCEEDINGS{ISSGM_ISM2020,
               AUTHOR    = "N. Gkalelis and V. Mezaris",
               TITLE     = "Structured Pruning of LSTMs via Eigenanalysis and Geometric Median for Mobile Multimedia and Deep Learning Applications",
               BOOKTITLE = "Proc. IEEE Int. Symposium on Multimedia (ISM)",
               ADDRESS   = "Naples, Italy",
               PAGES     = "",
               MONTH     = "Dec.",
               YEAR      = "2020"
}
```


## Acknowledgements

This work was supported by the EU Horizon 2020 research and innovation programme under grant agreement H2020-780656 ReTV.

## References

[1] K. Ota, M. S. Dao, V. Mezaris et al., “Deep learning for mobile multimedia: A survey,” ACM Trans. Mutlimedia Comput., Commun. and Appl., vol. 13, no. 3s, pp. 34:1–34:22, Jun. 2017.

[2] X. Suau, U. Zappella, and N. Apostoloff, “Filter distillation for network compression,” in IEEE WACV, Snowmass Village, CO, USA, Mar. 2020, pp. 3129–3138.

[3] Y. He, P. Liu, Z. Wang et al., “Filter pruning via Geometric median for deep convolutional neural networks acceleration,” in IEEE CVPR, Long Beach, CA, USA, Jun. 2019.

[4] N. Gkalelis and V. Mezaris, “Fractional step discriminant pruning: A filter pruning framework for deep convolutional neural networks,” in IEEE ICMEW, London, UK, Jul. 2020, pp. 1–6.

[5] W. Wen, Y. Chen, H. Li et al., “Learning intrinsic sparse structures within long short-term memory,” in ICLR, Vancouver, BC, Canada, Apr-May 2018.

[6] L. Wen, X. Zhang, H. Bai, and Z. Xu, “Structured pruning of recurrent neural networks through neuron selection,” Neural Networks, vol. 123, pp. 134–141, Mar. 2020.

[7] H. Xu, C. Caramanis, and S. Mannor, “Sparse algorithms are not stable: A no-free-lunch theorem,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 34, no. 1, pp. 187–193, Jan. 2012.
