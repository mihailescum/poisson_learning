## About

This repository contains all code related to my master's thesis on the continuum limits of Poisson learning, handed in October 2022 at the University of Bonn.

## Requirements

You will need to install the following packages:

* `python`(`3.9.7`)
* `numpy` (`1.12.2`)
* `pandas` (`1.3.4`)
* `scipy` (`1.7.2`)
* `graphlearning` (`1.1.3`)
* `matplotlib` (`3.4.3`)
* `seaborn` (`0.11.2`)
* `hdf5` (`1.12.1`)

## How to run

In the `examples` folder you find several preconfigured experiments that you can run. The results will be written to a `results` folder, which you should create in advance in your working directory. If you run the corresponding `<name>_results.ipynb` notebooks afterwards, the generated plots will be saved to a `plots` folder, which you should also create in advance.

For some of the experiments you can configure the number of threads to use using the `NUM_THREADS` variable at the begining of the scripts. Moreover, you have the possibility to specify a `SEED_RANGE`. Each value in this range will be the seed of a separate independent trial of the experiment you run, therefore you can test the experimental results for different, yet deterministic, inputs.

## Description of the experiments:

* `line`: 1D experiment for Poisson learning on the unit interval `(0, 1)` with two labeled nodes, one at `0.4`, one at `0.8`.
* `p_line`: Same as `line`, only that we do p-Poisson learning for a range of values of `p`.
* `one_circle`: 2D experiment for Poisson learning on the unit disc `B_1(0)` with two labeled nodes, one at `(-2/3, 0)` and one at `(2/3, 0)`.
* `p_one_circle`: Same as `one_circle`, only that we do p-Poisson learning for a range of values of `p`.
* `real_data`: p-Poisson learning on the two real world data sets `MNIST` and `FashionMNIST`.