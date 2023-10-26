FROM jupyter/datascience-notebook

RUN git clone https://github.com/gerritgr/moleculediffusiongan
RUN rm -rf moleculediffusiongan/.git 
#RUN cd dsnn && conda env update -f environment.yml -n base

RUN cd dsnn && mamba env create -f environment.yml -n moldiffgan
#RUN python -m ipykernel install --user --name=moldiffgan --display-name="Python (moldiffgan)"
RUN /opt/conda/envs/dsnnenv/bin/python -m ipykernel install --user --name=moldiffgan

RUN mamba clean -ya

RUN conda env export --name moldiffgan > environment_export.yml.txt # txt filending makes sure it is uploaded to wandb
RUN conda env export --name moldiffgan
