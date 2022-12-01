ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt update

RUN apt-get -y install aria2 nmap traceroute

# 3) install packages using notebook user
USER jovyan

# RUN conda install -y scikit-learn

RUN pip install pytorch 

RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

RUN pip install ogb

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]