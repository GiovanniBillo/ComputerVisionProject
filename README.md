# ComputerVisionProject

This repo contains all the files and documents for the project of the course 'Computer Vision and Pattern Recognition' by prof.Pellegrino at the University of Trieste.

# Description
The code for experiments is fully contained inside `run_bovw_experiments.py`.
In order to exeucte it without errors, please create an python environment beforehand with the `requirements.txt` file present.

```bash
python3 -m venv cvenv
source cvenv/bin/activate
pip install -r requirements.txt
```

# Instructions
We provide a shell script to run experiments on a computing cluster with SLURM.
To do this, simply execute the following:
```
sbatch cvpr_job.sh
```
This will launch a full evaluation and fit of all the models. Please beware that this can take some time to execute. 

