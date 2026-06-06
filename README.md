# HybridSplat-diff-surfel-rasterization

> **Modified version for [HybridSplat](https://github.com/Aetheryne/HybridSplat).**
> Original: [hbb1/diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization) via [EnvGS](https://github.com/zju3dv/EnvGS).

This is the 2D Gaussian Splatting rasterizer used by HybridSplat, extended with **reflection-sensitive importance scoring** to support the β-weighted gradient-pruning strategy.

## Modifications

### `scores` argument added to `GaussianRasterizer.forward()`

The original rasterizer signature:

```python
def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, ...)
```

is extended to:

```python
def forward(self, means3D, means2D, opacities, scores, shs=None, colors_precomp=None, ...)
```

`scores` is a per-Gaussian float tensor of shape `(N, 1)`. It is passed to the CUDA backend and accumulated during alpha compositing via the existing `out_weight` accumulation path. In the **backward pass**, the gradient with respect to `scores` — named `grad_gaussians2` in the original 2DGS convention — provides a differentiable importance signal per Gaussian.

In HybridSplat, this gradient is used to compute the **reflection-sensitive pruning score**:

```
Ũ_i = (1 - β) · (∇g_b · ∂I_G/∂G_base)²  +  β · (∇g_r · ∂I_G/∂G_ref)²
```

where the β weighting is realized by passing different `scores` tensors for the base and reflective branches.

### Package name

The package is named `diff_surfel_rasterization_wet_ch05` (ch05 = 5 output channels: RGB + depth + accumulated weight; wet = per-Gaussian accumulated weight). This name is carried over from the EnvGS codebase for import compatibility.

## Installation

```bash
pip install --no-build-isolation .
```

## Citation

If you use this rasterizer, please cite:

```bibtex
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    booktitle={SIGGRAPH 2024 Conference Papers},
    year={2024},
}

@article{xie2024envgs,
    title={EnvGS: Modeling View-Dependent Appearance with Environment Gaussian},
    author={Xie, Tao and Chen, Xi and Xu, Zhen and Xie, Yiman and Jin, Yudong and Shen, Yujun and Peng, Sida and Bao, Hujun and Zhou, Xiaowei},
    journal={arXiv preprint arXiv:2412.15215},
    year={2024},
}
```
