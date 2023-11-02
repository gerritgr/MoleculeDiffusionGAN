FROM jupyter/datascience-notebook

RUN git clone https://github.com/gerritgr/moleculediffusiongan
RUN rm -rf moleculediffusiongan/.git 

RUN jupyter nbconvert --to script --output "moleculediffusiongan/main.py" "moleculediffusiongan/main.ipynb"

RUN cd moleculediffusiongan && mamba env create -f environment.yml -n moldiffgan

RUN /opt/conda/envs/moldiffgan/bin/python -m ipykernel install --user --name=moldiffgan

RUN mamba clean -ya

RUN conda env export --name moldiffgan > environment_export.yml.txt # txt filending makes sure it is uploaded to wandb
RUN conda env export --name moldiffgan
