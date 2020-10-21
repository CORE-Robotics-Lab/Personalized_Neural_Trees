# Interpretable and Personalized Apprenticeship Scheduling: Learning Interpretable Scheduling Policies from Heterogeneous User Demonstrations

This is the codebase for creating Personalized Neural Trees (PNTs) and generating
performance results in each domain including the low-dim, synthetic scheduling,
and Taxi domain.

### Requirements

Requirements are included in the `requirements.txt` file, and this repo itself is a requirement. Install by running the following in the main directory:
```
$ pip install -r requirements.txt
$ pip install -e .
```
### Navigating the Repo
![Results](https://github.com/Personalized-Neural-Trees/Interpretable-and-Personalized-Apprenticeship-Scheduling-Learning-Interpretable-Scheduling-Policies/blob/master/results_img.png)


The table below relates author names within the results above to filenames.

| Method Name                | FileNAme          |
| -------------------------- | ----------------- |
| Our Method                 | PNT               |
| Sammut et. al.             | NN                |
| Nikolaidis et. al.         | k-means_NN        |
| Tamar et. al.              | SNN               |
| Hsiao et. al.              | cVAE              |
| Gombolay et. al.           | DT_pairwise       |
| InfoGAIL                   | InfoGAIL          |
| Our Method (interpretable) | Crisp             |
| Our Method (DT+\omega)     | DT_our_embeddings |

### Low-Dim

---------------------------

For this environment, the ```generate_environment.py``` file is in charge of creating a dataset. The datasets folder has several datasets we created and train/test upon.

Each of the methods are found within this subfolder, corresponding to algorithm names within their respective papers.

### Scheduling

---------------------

For this environment, generating successful schedule trajectories takes some time. The folder ```create_scheduling_data``` has the files necessary to generate large datasets. Running create_data.py can create any number of schedules, averaging about 1000 a day.

As these datasets are large, we have not added ours. However, as we use seeds, generating 250 schedules will produce the same dataset as ours. 

Each of the methods are found within this subfolder, corresponding to algorithm names within their respective papers. Note that each algorithm will run several times for k-fold cross` validation, sampling a different subset of 150 schedules to train upon.

### Taxi

For this environment, data generation is done through ```taxi_sim.py```. This file is the images in the folder ```taxi_tree_images``` represented as if statements and generates trajectories by sampling from users. 

We have several datasets creating by randomly sampling demonstrators and initial states. Permuting them for training and testing gives several accuracy measures we use to compute the k-fold cross validation accuracy reported within the paper.

Each of the methods are found within this subfolder, corresponding to algorithm names within their respective papers.

### Implementation tips when using PNTs

- Pretraining the policy before starting variational inference typically leads to
better performance and stability during training.
- Discretization is very sensitive and does not have a simple relationship with policy performance.
