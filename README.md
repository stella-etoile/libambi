# libambi: A Proof-of-Concept Learned Image Compression Library
## Introduction

libambi is a proof-of-concept research project exploring the potential of learned image compression. Unlike traditional codecs (JPEG, PNG, WebP, AVIF), which rely on hand-designed transforms and heuristics, libambi learns compression strategies directly from data.

This project has been lightly trained on ~200 GB of images — far less than state-of-the-art research systems — but still demonstrates the viability of this approach. It is not intended as a production-ready codec, but rather as a foundation for experimentation, theory testing, and future research in adaptive compression.

The goal is to show how reinforcement learning–based optimization of rate–distortion tradeoffs can adaptively outperform rigid, pre-defined codecs when given enough data and compute.

## Why Learned Compression?

Traditional codecs (JPEG, JPEG2000, WebP, AVIF) are built on fixed transforms (DCT, wavelets, etc.), hand-tuned quantization tables, and universal entropy models. While efficient and widely supported, they suffer from several limitations:

Rigid tradeoffs: You pick a “quality” value, but the mapping to bitrate/distortion is not flexible.

Weak adaptability: The same quantization is applied across very different images, leading to inefficiencies.

Suboptimal transforms: Fixed frequency transforms can’t fully exploit learned correlations in natural images.

Learned compression, by contrast:

Optimizes directly for data: Models learn priors from real images, adapting to distributional statistics.

Flexible tradeoffs: Reinforcement learning rewards can be tuned to prioritize bitrate, PSNR, or perceptual quality.

End-to-end training: Quantization, entropy coding, and distortion are optimized jointly.

The result is a system that, given sufficient training data, achieves a better rate–distortion curve than classical codecs — compressing more efficiently while preserving perceptual quality.

## Theoretical Foundation

Image compression can be formulated as a rate–distortion optimization problem: the objective is to minimize the number of bits required to represent an image (rate) while minimizing the difference between the reconstructed and original image (distortion).

Traditional codecs such as JPEG and WebP approach this problem using fixed transforms and quantization. These methods are equivalent to applying predetermined interpolation or approximation rules that do not adapt to the distribution of natural images. While effective in simple cases, they fail to capture higher-order dependencies and often allocate bits inefficiently.

In contrast, learned compression replaces fixed rules with a statistical model that predicts image content given context. The closer this prediction is to the true distribution of natural images, the fewer residuals need to be encoded, and thus the lower the bitrate for a given quality.

The reinforcement learning (RL) component in libambi governs how the codec learns this predictive process. The policy is trained with a reward function that balances:

Rate (R): the expected number of bits per pixel.

Distortion (D): a measure of reconstruction error, using metrics such as PSNR, SSIM, or MS-SSIM.

The RL agent learns to select compression strategies that minimize rate while maintaining acceptable distortion. Unlike fixed interpolation, the policy adapts its decisions to local image statistics. This adaptivity leads to systematically better predictions of image structure, resulting in smaller residuals and therefore improved rate–distortion performance.

Because the system optimizes prediction directly on natural images, it learns correlations and dependencies that hand-crafted transforms cannot capture. Even with limited training on ~200 GB of data, the model demonstrates the feasibility of surpassing traditional codecs by learning to allocate bits more efficiently and reconstruct structure with higher fidelity.