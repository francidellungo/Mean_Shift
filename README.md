# Mean Shift clustering
[Mean shift](https://en.wikipedia.org/wiki/Mean_shift) clustering algorithm is an unsupervised clustering algorithm; in particular it is a non-parametric clustering  method based on kernel-density estimation.
It aims to discover “blobs” in a smooth density of samples. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region.

<p align="center">
  <img src="https://github.com/francidellungo/Mean_Shift/blob/master/readme_imgs/mean-shift.gif" width="600">
</p>

### Directories Layout

```bash
├── dataset                    # Generated datasets
│   ├── 3d                     # 3d points (x,y,z)
│   │   ├── 100                # Different sizes
│   │   ├── 1000 
│   │   ├── 10000 
│   │   ├── ... 
├── experiments                # Experiments results
│   ├── current_time_0         # First set of experiments
│   │   ├── seq        
│   │   ├── openmp        
│   │   ├── cuda     
│   ├── current_time_1 
│   │   ├── seq        
│   │   ├── openmp        
│   │   ├── cuda  
├── ...
```

## Dataset
Create new folder: `mkdir dataset` 

Create sub-folders: `mkdir dataset/variable_size`

## Experiments
Create new folder: `mkdir experiments` 


## Authors
This project was carried out in collaboration with [Matteo Petrone](https://github.com/matpetrone) for the Parallel Computing exam.

## WORK IN PROGRESS...
