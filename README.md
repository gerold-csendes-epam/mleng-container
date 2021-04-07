# Containerization (Docker)
This is my assignment for the first course task in the Machine Learning Engineering track. 

List of utilized resources: 
- https://medium.com/swlh/resnet-with-tensorflow-transfer-learning-13ff0773cf0c

## Using docker
Run the following command to get docker up and running: 
```bash
docker container run -p 8888:8888 mleng-container
```

Or run this comment to mount data

```bash
docker container run -p 8888:8888 --mount source=mleng-data,destination=/home/jovyan/data/ mleng-container
```

Now enter **localhost:8888** in your browser of choice.
