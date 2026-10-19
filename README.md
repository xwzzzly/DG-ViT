# DG-ViT

DG-ViT code for fusing multi-spaceborne GNSS-R observations for soil moisture retrieval over the Tibetan Plateau, accompanying our manuscript under review at JAG.

This repository contains the DG-ViT model and land cover informed prior weight estimation code. 
## Scope

Included:

- dgvit/model.py: source embeddings, Dirichlet gating, Transformer encoder, and gated residual reconstruction.
- dgvit/priors.py: land cover class mapping and prior-weight loading and normalization.
- estimate_priors.py: land cover informed prior weight estimation using inter-mission consistency, coverage, and temporal stability.
- train.py and infer.py: model training and daily fusion.
- compute_normalization.py and build_manifest.py: utilities for users to prepare their own model inputs.
- Supporting configuration, data-loading, raster, loss, metric, and model-factory modules under dgvit/.


## Inputs and configuration

The source order is CYGNSS, Tianmu-1, FY-3E, FY-3F, FY-3G. Users must supply their own quality-controlled SR rasters, date manifest, land cover raster, and estimated prior-weight CSV. SR inputs are represented in dB.

configs/default.json contains method and runtime parameters only. Its normalization means are zero and standard deviations are one as placeholders, not fitted statistics. Estimate normalization from the intended training data before formal training; do not interpret these placeholders as the parameters used to obtain published results.

## Implementation

Prior scores use non-negative correlation values, exponentially decreasing bias and ubRMSE scores, coverage and temporal stability. The prior-estimation implementation handles zero-score cases explicitly.

The Dirichlet concentrations are positive. Initial weights are sampled during training; normalized concentrations are used during inference. Both are adjusted for source availability. The model combines the resulting SR baseline with a gated residual correction.

## Usage

Install the dependencies listed in requirements.txt in a suitable Python environment. Each entry point provides command-line help through --help.

A typical workflow is to prepare a date manifest, compute training-data normalization, estimate land cover informed prior weights, train the model, and run inference. All user data and generated files should be kept outside the source repository.

No new algorithm changes were made when preparing this source-only copy. No new training or end-to-end experiment was run. Selection of a public-release license and final documentation will be handled after the owner decides to publish the repository.
