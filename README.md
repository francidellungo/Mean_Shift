# Mean Shift clustering
[Mean shift](https://en.wikipedia.org/wiki/Mean_shift) clustering algorithm is an unsupervised clustering algorithm; in particular it is a non-parametric clustering  method based on kernel-density estimation.
It aims to discover “blobs” in a smooth density of samples. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region.

<p align="center">
  <img src="https://github.com/francidellungo/Mean_Shift/blob/master/docs/readme_imgs/mean-shift.gif" width="600">
</p>

### Directories Layout

```bash
├── docs                       # Documentation files
│   ├── readme_imgs
│   ├── figures
│   ├── MeanShift_paper        # article for the exam 
│   ├── MeanShift_presentation # slides for the exam 
├── dataset                    # Generated datasets
│   ├── 3d                     # 3d points (x,y,z)
│   │   ├── 100                # Different sizes
│   │   ├── 1000 
│   │   ├── 10000 
│   │   ├── ... 
├── CUDA                       # Cuda files
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
Create new folder: 
 
 `mkdir dataset` 

Create sub-folder:
 
 `mkdir dataset/3d`

Dataset can be generated using 'generateDatasets()' function inside the generatePoints.py file. 
This will automatically generate 9 sets of 3d points, saved in the dataset directory.

## Experiments
Create new folder:

`mkdir experiments` 

## Run experiments 
### C++ and openMP
 `main.cpp` contains the experiments for the sequential version implemented in C++ and for the parallel version with openMP.
 If you want to change the datasets to work with please refer to `dimensions` variable inside the main function in the same file.
Before compile and run verify to have all the dataset files in the directories and the CMake file correctly set.

If you want to compile and run by cmd, you can compile with the following command:

`g++ -std=c++14 -fopenmp Point.cpp Point.h main.cpp Cluster.cpp Cluster.h Utils.cpp Utils.h MeanShift.cpp MeanShift.h MeanShiftUtils.cpp MeanShiftUtils.h Tests.cpp -o main` 

and then run with:

`./main`


### CUDA 
By default are launched experiments for the CUDA naive version and for the CUDA tiling version for all the dataset specified in 
`dimensions`  array in file MeanShift.cu. If you want to change the CUDA version to execute is sufficient to change the values of `tiling_experiments` in the same file.
Assures to have all the datasets files, all the directories created and the CMake file correctly set before starting the execution. 

If you want to compile and run the cuda version by cmd, you can compile with the following command:

`nvcc CUDA/MeanShift.cu -o MeanShift` 

and then run with:

`./MeanShift`

## Authors
This project was carried out in collaboration with [Matteo Petrone](https://github.com/matpetrone) for the Parallel Computing exam.

## WORK IN PROGRESS...
