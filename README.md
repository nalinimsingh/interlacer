# Interlacer
Joint frequency- and image- space learning for Fourier imaging tasks.

![Sample reconstruction results](./assets/teaser.jpg)

keywords: image reconstruction, motion correction, denoising, magnetic resonance imaging, deep learning

## Dependencies
All dependencies required to run this code are specified in `requirements.txt`. To create an anaconda environment with those dependencies installed, run `conda create --name <env> --file requirements.txt`. You will also need to add this repo to your python path (if you're using conda, `conda-develop /path/to/interlacer/`).

## Layer Implementation
If you'd like to incorporate our joint learning strategy into your own networks, we provide a standalone Keras Layer in `interlacer/layers.py`. This layer currently supports only 2D inputs.

## Training Code
Unfortunately, we are unable to provide the images used for training, due to license restrictions. However, we provide code to train on alternative datasets. To specify your own dataset paths and paths for output of training results, fill in the appropriate fields in `scripts/filepaths.py`.

The entry to our training code is in `scripts/train.py`, which is called via `python scripts/train.py $path_to_config.ini`. Running this script:
* reads the experiment configuration specified
* loads the appropriate model architecture 
* loads the training data
* executes training
* writes all training logs to a subdirectory created under `training/`.

We provide a helper script to generate config files for experiments comparing multiple models in `scripts/make_configs.py`. This script allows the user to specify the name of an experiment as well as lists of model/data parameters to be tried (e.g. a list of model architectures). Running `scripts/make_configs.py` creates the subdirectory `configs/$experiment_name`, which contains a single configuration file for each specified model/data combination. 

For SLURM users, running `python scripts/run_experiment.py ../configs/$experiment_name$` starts training (by running `train.py`) for each configuration file within the directory.

## Docker

### Create Docker ENV
```
docker build -t interlacer/base -f .\docker\Dockerfile.base .
```

### Create module tests ENV
```
docker build -t interlacer/tests -f .\docker\Dockerfile.tests .
```

### Run tests
```
docker run interlacer/tests
```

out:
```
(base) PS C:\Users\Drmis\Desktop\projects\interlacer> docker run interlacer/tests
....
----------------------------------------------------------------------
Ran 4 tests in 0.001s

OK
```

> remove none tag images  
> docker rmi $(docker images --format '{{.ID}}' --filter=dangling=true) -f

### Start train with Docker
Your local configs folder is connected and the necessary configuration file is taken from there.

```
docker run -it --volume $pwd/configs:/usr/src/app/configs --volume $pwd/training:/usr/src/app/training  interlacer/tests python scripts/train.py configs/path_to_config.ini
```

### images
```
    (base) PS C:\Users\Drmis\Desktop\projects\interlacer> docker images
    REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
    interlacer/tests    latest              d237e73a72d6        2 minutes ago       7.64GB
    interlacer/base     latest              ce6e6d21771b        About an hour ago   7.64GB
    conda/miniconda3    latest              2c4c668a3586        15 months ago       355MB
```



## Paper 
If you use the ideas or implementation in this repository, please cite our [paper](https://arxiv.org/abs/2007.01441):

    @misc{singh2020joint,
        title={Joint Frequency- and Image-Space Learning for Fourier Imaging},
        author={Nalini M. Singh and Juan Eugenio Iglesias and Elfar Adalsteinsson and Adrian V. Dalca and Polina Golland},
        year={2020},
        eprint={2007.01441},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }       
