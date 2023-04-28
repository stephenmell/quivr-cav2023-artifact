# Quivr: Artifact for "Synthesizing Trajectory Queries from Examples" at CAV 2023

## Experiments
We evaluate on three datasets, for a total of 17 benchmark tasks.

### F1
In the paper, this is Table 1. We compare the F1 scores, on each task, of:
- our approach (Q)
- an ablation of our approach which makes random labeling choices (R)
- logistic regression on an LSTM embedding of trajectories (L)
- a transformer (T)

### Timing
In the paper, this is Table 2. We make two comparisons:
- our approach (Q) versus an ablation using binary search rather than quantitative semantics (B)
- executing on a CPU versus on a GPU
We consider all four combinations: (B, CPU), (Q, CPU), (B, GPU), (Q, GPU)

## Dependencies
numpy, pytorch, and scikit-learn

For CPU and CUDA, `conda.yaml` contains an exported Conda environment with the necessary packages, and `Dockerfile` will produce a Debian image using this Conda environment.

To use Docker/Podman:
```
docker build -t quivr .
docker run -v "$(realpath .)":/root/project -w /root/project --rm -ti quivr {command}
```
It can be tricky to configure Docker/Podman to support CUDA, and it may be easier to use Conda directly. If CUDA is supported, then this should list your GPUs:
```
docker run -v "$(realpath .)":/root/project -w /root/project --rm -ti quivr nvidia-smi
```

To use Conda directly:
```
conda env create -f conda.yaml -n quivr
conda activate quivr
```

The Conda environment `conda.yaml` was produced with Miniconda 23.1.0, and then:
```
conda install numpy
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install scikit-learn
```

It should be possible to use AMD GPUs by installing PyTorch ROCm in pip, along with numpy and scikit-learn.

## Reproducing Results
Evaluating on a CPU is quite slow, so if at all possible we recommend using a GPU for the F1 results. All experiments require less than 3.5 GB of GPU memory.

For all commands, the working directory should be the root of this repository.

The scripts currently do not do checkpointing, so it is advisable to run them in something like `screen` or `tmux`.

The scripts accept a list of tasks, of which `all` and `smoketest` are provided (see `task_lists/`). `smoketest` should run quickly, whereas `all` may take a long time.

Download the dataset tarball (DOI 10.5281/zenodo.7872795) and extract it into the root of this repository:
```
wget https://zenodo.org/record/7872796/files/quivr_dataset.tar.xz
tar -xJvf quivr_dataset.tar.xz
```

If "device" is a GPU (e.g. "cuda:0"), it will run much more quickly, but "cpu" will work in principle. For the timing experiments, a GPU is required for "gpu_device", or "skip" can be used to skip the GPU part of the experiment. "cpu_device" should be "cpu", or "skip" can be used to skip the CPU part of the experiment.

To generate the LaTeX F1 table (Table 1):
```
python run_f1.py {task_list} {device}
python quivr/table_f1.py {task_list} {device}
```
On an NVIDIA GeForce RTX 2080 Ti (a 2018 desktop GPU), running all F1 experiments takes about 12 hours and generating the table takes about 10 minutes.

To generate the LaTeX timing table (Table 2):
```
python run_timing.py {task_list} {cpu_device|skip} {gpu_device|skip}
python quivr/table_timing.py {task_list} {device}
```
On an NVIDIA GeForce RTX 2080 Ti (a 2018 desktop GPU), running the GPU portion of all timing experiments takes (for 3 trials) about 9 hours. On an Intel Core i7-6700 CPU @ 3.40GHz (a 4-core/8-thread 2015 desktop CPU), running the CPU portion of the all timing experiments takes (for 3 trials) about 30 hours hours, while smoketest portion takes (for 3 trials) about 5 minutes.

## Running on New Datasets
Add the dataset to `datasets/{dataset_name}` and the labels to `datasets/{dataset_name}/task_{task_name}/task.torch` in the formats specified below, and then run:
```
quivr/do_active_learning.py {dataset_name} {task_name} quantitative no {torch_device} smart
```

The learned queries can be found in `quivr_results/{dataset_name}/task_{task_name}/al_state_smart_{al_step}.torch`.

## File Formats
`task_lists/{all, smoketest}` (JSON)
A list of triples, (dataset_name, task_name, task_id), where task_id is just used for display purposes in the LaTeX tables.

`datasets/{dataset_name}/data.torch` (`torch.load`)
```
{
    'traces': {
        '{predicate_name}': torch tensor with
                            dimensions [n, m, m] where n is the number of traces in the dataset and m is the number of time steps per trace
                            dtype bool (for predicates with no parameters) or float32 (for predicates with parameters)
    },
    'pred1_bounds' :{
        '{predicate_name}': tuple of (lower bound, upper bound) where both are floats
                            only predicates with parameters occur here
    },
    'train_indices': torch int64 tensor with the indices (in 'traces') of the training examples
    'test_indices': torch int64 tensor with the indices (in 'traces') of the test examples
}
```

`datasets/{dataset_name}/task_{task_name}/task.torch` (`torch.load`)
```
{
    'labels': torch bool tensor with dimensions [n] where n is the number of traces in the dataset
    'labeled_initial_indices': torch int64 tensor with the indices (in 'traces') of the initial
                               labeled examples (should be a subset of 'train_indices')
}
```

`quivr_results/{dataset_name}/task_{task_name}/al_state_{smart|random}_{al_step}.torch` (`torch.load`)
```
{
    'unpruned_queries': list of tuples of the form: (expr, sat, unk) where
                        'expr' is a sketch
                        'sat' is a list of boxes in parameter space (tensors)
                        'unk' is a list of boxes in parameter space (tensors)
                        (if 'sat' is nonempty, we know that there is a parameter
                         satisfying the labeled data)
    "labeled_indices": torch int64 tensor with the indices (in 'traces') of the labeled examples
                       so far (should be a subset of 'train_indices')
}
```
