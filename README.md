# 3D Motion Segmentation with SSC

This repository implements a SSC-based motion segmentation algorithm on 2-frame point cloud (scene flow).
The implementation basically follows this paper [3D Motion Segmentation of Articulated Rigid Bodies based on RGB-D Data](http://bmvc2018.org/contents/papers/0901.pdf).

# Data preparation

The implemented algorithm is tested on synthetic data and real data.

## Synthetic data

Please play with [prepare_toy_data.py](prepare_toy_data.py) to generate synthetic datasets.

## Real data

The algorithm is tested on 2 public scene flow datasets: FlyingThings3D and KITTI.
The processed data is available at [real_data.zip](real_data.zip)

You can also download the source data and process it by yourself.
Please first follow the guidelines in [flowstep3d](https://github.com/yairkit/flowstep3d) to download and preprocess the data.
Then use the provided [prepare_real_data.py](prepare_real_data.py) for further processing.

# How to use

On toy data

```
python eval_toy_data.py --data_path <Path to data> --n_cluster <GT number of clusters> --normalize
```

On real data

```
python eval_real_data.py --data_path <Path to data> --normalize --min_n_cluster <Minimum number of clusters when self-tune> --max_n_cluster <Maximum number of clusters when self-tune>
```

# Performance

## Qualitative results

On FlyingThings3D

![](eval_results/ft3d_samples.png)

On KITTI

![](eval_results/kitti_samples.png)

(* Each scene in FT3D / KITTI is subsampled into 1024 points to speed-up)

## Quantitative results

On FlyingThings3D

| Dataset | SSR error | Clustering error | AP@50 | Precision@50 | Recall@50 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| FT3D | 0.0290 | 0.3723 | 0.5060 | 0.7644 | 0.2476 |
| FT3D (>0.01) | 0.0259 | 0.3592 | 0.5649 | 0.7773 | 0.3524 |
| FT3D (>0.02) | 0.0256 | 0.3151 | 0.6413 | 0.8196 | 0.4630 |

(* For FT3D, only first 200 samples in test set are evaluated)

On KITTI

| Dataset | SSR error | Clustering error | AP@50 | Precision@50 | Recall@50 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| KITTI | 0.0078 | 0.0792 | 0.8309 | 0.9038 | 0.7580 |
| KITTI (>0.01) | 0.0078 | 0.0748 | 0.8411 | 0.9141 | 0.7681 |
| KITTI (>0.02) | 0.0076 | 0.0650 | 0.8734 | 0.9373 | 0.8094 |

(* > 0.01/0.0.2 means objects with too few points are removed)

# Acknowledgements

[sparse-subspace-clustering-python](https://github.com/abhinav4192/sparse-subspace-clustering-python)

[STSC](https://github.com/wOOL/STSC)

[flowstep3d](https://github.com/yairkit/flowstep3d)