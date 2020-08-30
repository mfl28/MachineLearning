FROM jupyter/base-notebook
COPY requirements.txt /home/jovyan/work
WORKDIR /home/jovyan/work
RUN sed -i /jupyter/d requirements.txt &&  pip install -r requirements.txt 
