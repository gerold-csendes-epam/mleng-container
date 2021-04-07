FROM jupyter/base-notebook:python-3.8.8

LABEL maintainer="Gerold Csendes <gerold_csendes@epam.com>"
LABEL description="MLeng assignment to create dev env for an ML project"

COPY /notebooks notebooks/
COPY requirements.txt .
# COPY /models models/

# VOLUME /data

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["jupyter","notebook","--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"] 
