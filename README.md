# MoleculeDiffusionGAN
<div style="text-align:center;">
<img src="method.jpg" alt="alt text" title="Overview" width="700"/>
</div>
WIP implementation of a discriminator-guided diffusion model for molecule generation. 

This repository provides a proof-of-concept implementation for the manuscript _[Discriminator-Driven Diffusion Mechanisms for Molecular Graph Generation](https://github.com/gerritgr/MoleculeDiffusionGAN/blob/73993c20db724a6c0ceb9f6f29ce8092141b58c1/Discriminator_Guided_Diffusion_for_Molecule_Generation.pdf)_.



## Results
<div style="text-align:center;">
<img src="results.jpg" alt="alt text" title="Overview" width="400"/>
</div>

## Run MoleculeDiffusionGAN 

### On Colab
You can run the _main.ipynb_ on Google Colab following [this URL](https://colab.research.google.com/github/gerritgr/MoleculeDiffusionGAN/blob/main/main.ipynb). 

### Locally

You can run DSNN locally using _main.ipynb_. First, install Anaconda, then create an environment with the Python dependencies (tested on _OS X_):

```console
conda env create -f environment.yml -n moldiffgan
conda activate moldiffgan
jupyter lab
```
Then just run the notebook from start to finish. 

### Via Docker
Install docker and then:
```console
docker pull gerritgr/moleculediffusiongan:latest
docker run -p 8888:8888 gerritgr/moleculediffusiongan:latest
```
Next:
1) Manually copy the URL to your browser (if other instances of jupyter lab are running, this can lead  to problems).
2) Navigate to the notebook
3) Activate the _moldiffgan_ kernel (Kernel -> Change Kernel -> select _moldiffgan_).
4) Run _main.ipynb_ from start to finish. 
