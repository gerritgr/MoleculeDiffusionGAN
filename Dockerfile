FROM jupyter/datascience-notebook

RUN git clone https://github.com/gerritgr/dsnn
RUN rm -rf dsnn/.git 
#RUN cd dsnn && conda env update -f environment.yml -n base

RUN cd dsnn && mamba env create -f environment.yml -n dsnnenv
#RUN python -m ipykernel install --user --name=dsnnenv --display-name="Python (dsnnenv)"
RUN /opt/conda/envs/dsnnenv/bin/python -m ipykernel install --user --name=dsnnenv

RUN mamba clean -ya

RUN conda env export --name dsnnenv > environment_export.yml.txt # txt filending makes sure it is uploaded to wandb
RUN conda env export --name dsnnenv
