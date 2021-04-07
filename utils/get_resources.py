import pathlib
import os
import numpy as np
from keras import datasets, layers, models, losses, Model
from keras.applications import ResNet152
from keras.callbacks import EarlyStopping

def main():

    # load data
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    script_path = pathlib.Path(__file__).resolve()
    root_path = script_path.parent.parent

    # save data
    data_dir = root_path.joinpath("data", "raw")
    np.save(data_dir.joinpath(data_dir, "X_train"), X_train)
    np.save(data_dir.joinpath(data_dir, "y_train"), y_train)
    np.save(data_dir.joinpath(data_dir, "X_test"), X_test)
    np.save(data_dir.joinpath(data_dir, "y_test"), y_test)

    # load resnet 152 and save

if __name__ == "__main__":
    main()
