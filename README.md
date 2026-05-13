# LiDAR View Synthesis — ASAR Lab

Undergraduate research project benchmarking interpolation methods for LiDAR view synthesis using an Ouster OS2-128 sensor. Research conducted at ASAR Lab, Tufts University, under Professor Jason Rife (2025–2026).

## Overview

Given two LiDAR frames from different positions, the goal is to synthesize what the scene looks like from the reference frame using data from a new frame and to compare any two frames regardless of movement transform. This is evaluated by computing RMSE between the synthesized range image and the ground truth reference.

Methods compared:
- **Bilinear Splat with Gaussian Convolution** — bilinear splatting onto oversampled grid with Gaussian spectral hole filling
- **Bilinear Splat** — bilinear splatting only, no hole filling
- **Nearest Neighbor** — nearest neighbor via `scatteredInterpolant`
- **Linear** — linear interpolation via `scatteredInterpolant`
- **FFT Lookup** — fractional IDFT evaluated at transformed reference pixel positions in the new frame's frequency domain

## Repository Structure

```
lidar-reconstruction-research/
├── analysis/
│   ├── LidarCompare.m      — single frame pair visualization and method exploration
│   ├── MeanError.m         — single dataset RMSE vs pose distance
│   ├── Distributions.m     — single frame pair error distribution histograms
│   ├── rmse.m              — full benchmark all methods
│   ├── local_rmse.m        — local methods 
│   ├── fft_store.m         — extracts FFT results (long runtime to save FFT method results)
│   └── rmse_graph.m        — pools all datasets, plots smoothed RMSE vs pose distance
├── io/
│   └── VolpeOusterRead.m   — reads raw .pcap files and writes .mat clips
├── docs/
│   └── LiDAR View Synthesis Poster - Rife 2026.pdf
└── data/                   — .mat clip files (not tracked, see Data section)
```

## Workflow

```
Single frame exploration  →  LidarCompare.m, Distributions.m
Single dataset sweep      →  MeanError.m
Full benchmark            →  rmse.m or local_rmse.m
Extract FFT results       →  fft_store.m
Plot across datasets      →  rmse_graph.m
```

## Data

Raw `.pcap` and processed `.mat` files are not included in this repository due to size. Data was collected March 2025 at Volpe National Transportation Systems Center using an Ouster OS2-128 (2048×10 mode).

To reproduce results, place `.mat` data files in `data/`.

## Dependencies

- MATLAB R2022b or later
- Computer Vision Toolbox (`pcregistericp`, `pointCloud`)
- Ouster MATLAB SDK (`ousterFileReader`, `readFrame`)

## Known Limitations

- FFT Lookup assumes signal periodicity across the azimuth axis; produces Gibbs ringing at depth discontinuities
- Ego mask coordinates are hardcoded for this specific vehicle and mount configuration
- ICP registration quality degrades at large pose distances

## Future Work

- Preprocessing data in frequency domain first to be able to better generalize across scenes
- Extension to multi-frame reconstruction using multiple source views
- Evaluation on additional datasets beyond Volpe collection

## Contact

Wilson Wu — wilson.wu@tufts.edu  
