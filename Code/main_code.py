# General imports
import os
import datetime
import numpy as np
import random

import yaml
import cv2
from PIL import Image
import argparse

# Importing torch and the related packages
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# Importing torch vision model
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
# from torchvision.io import read_image
import torchlars as LARS

# Importing support packages
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Importing custom files
import Model_resnet
import Dataloader
import Model_resnext
import train_loop
import suppliment
import Model_Custom
import EfficientNet_model

from sklearn.model_selection import KFold

# Printing the pytorch and torch vision version.
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# parser = argparse.ArgumentParser(description='Process flags as input')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')

parser = argparse.ArgumentParser(description='Process flags as input')
parser.add_argument("--config",
                    nargs="?",
                    const=r"Config/Config_basic.yaml")
args = parser.parse_args()

if args.config is None:
    print("No input was detected")
    # config_file = "./config/Config_Test.yaml"
    config_file = r'Config/Config_basic.yaml'
else:
    config_file = args.config

if __name__ == '__main__':
    # Reading config data
    with open(config_file, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful")

    global step, ptc_close, loss_func_name, val_step, test_flag, pretrain, model_structure, device
    global show_image_size, writer, model_version, Debug
    step = 0
    val_step = 0
    init_epoc = 0
    #  Adding seed to get some re-train-ability
    seed = int(data['seed'])
    torch.manual_seed(seed)

    # Data Source
    data_source = data['csv_source']
    normalize = bool(data['normalize'])
    k_cross = int(data['k-cross'])

    # Test Flags
    test_flag = bool(data['test_flag'])
    check_dataLoad = bool(data['check_dataloader'])
    pretrain = bool(data['preTrain'])
    show_image_size = bool(data['show_image_size'])

    # hyper parameters
    loss_func_name = data['loss_function']
    batch_size = int(data['batch_size'])
    learning_rate = float(data['learning_rate'])
    epochs = int(data['epochs'])
    pct_close = float(data['pct_close'])
    optimizer_type = data['optimizer']

    # Current model - architecture being run
    model_structure = data['model_structure']
    model_version = data['resnet']
    model_savePath = data['model_path']

    # Debug Mode
    Debug = bool(data['Debug'])

    # Variables
    recover = bool(data['program_complete'])
    last_save = data['last_save']
    last_epoch = int(data['last_epoch'])

    # Deciding which device to used
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('Using {} device'.format(device))

    if test_flag:
        epochs = 5

    # Creating a dynamic tensorboard filename
    datetime_object = datetime.datetime.now()
    date_time_clean = datetime_object.strftime('%m%d%Y_%H%M%S')
    run_name = f'DataFlattening_{date_time_clean}_{model_structure}_{model_version}_{loss_func_name}_{batch_size}_' \
               f'{learning_rate}_{epochs}_{normalize}'

    # Defining the tensorboard data writer.
    if not Debug:
        writer = SummaryWriter(os.path.join(r'D:\Machine Learning\Projects\DataFlattening\TensorBoard', run_name))
    else:
        writer = 'SampleWriters'
    # Initializing all the variables to the files
    train_loop.init(step, loss_func_name, val_step, device, writer, normalize, Debug)
    suppliment.init(pct_close)
    Dataloader.init(show_image_size)

    # Defining the models and initialing the required model
    Model_resnet.init(pretrain, model_version)
    Model_Custom.init(pretrain, model_version)
    EfficientNet_model.init(pretrain, model_version)
    Model_resnext.init(pretrain, model_version)

    # Defining the different loss functions for the model
    if loss_func_name == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()  # softmax output with n classes
    elif loss_func_name == 'L1loss':
        loss_fn = nn.L1Loss()
    elif loss_func_name == 'SmoothL1loss':
        loss_fn = nn.SmoothL1Loss()
    elif loss_func_name == 'MSEloss':
        loss_fn = nn.MSELoss()
    elif loss_func_name == 'RMSEloss':
        loss_fn = train_loop.RMSELoss()
    else:
        loss_fn = nn.L1Loss()

    columns = ['Data', 'Yield_Data']
    try:
        data_df = pd.DataFrame(np.load(data_source, allow_pickle=True), columns=columns)
    except IOError:
        print("File was unable to read.")
        data_df = pd.DataFrame(np.load('data.npy', allow_pickle=True), columns=columns)

    min_max = pd.Series(data=[0, 0], index=['max', 'min'])
    if normalize:
        # min_max = suppliment.minMax(data_df['Yield_Data'])
        min_max['min'], min_max['max'] = 0, 150
        print(f'The minimum and maximum values of yield are : {min_max}')
        data_df['Yield_Data'] = data_df['Yield_Data'].apply(suppliment.normalized,
                                                            args=(min_max['min'], min_max['max']))

    if test_flag:
        data_df = data_df.head(1000)

    # Defining the transform
    transform = transforms.Compose([
        # resize - not possibleunless the image is a square and there will be data loss.
        # transforms.Resize(32),
        # to-tensor
        transforms.ToTensor(),
        # normalize - not reuqired if the values has already been normalized
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # train_df, test_df = train_test_split(data_df, test_size=0.2)
    # train_df = train_df.reset_index()
    # test_df = test_df.reset_index()

    kf = KFold(n_splits=k_cross)
    kf.get_n_splits(data_df)
    X_train, X_test = [], []
    for train_index, test_index in kf.split(data_df):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train.append(data_df.loc[train_index])
        X_test.append(data_df.loc[test_index])

    data_change_epoch = int(epochs / k_cross)

    # train_df, test_df = X_train[0], X_test[0]
    # train_df = train_df.reset_index()
    # test_df = test_df.reset_index()
    #
    # # Defining the dataset from the generated Dataframe
    # train_dataset = Dataloader.CustomImageDataset(annotations_file=train_df, transform=transform)
    # test_dataset = Dataloader.CustomImageDataset(annotations_file=test_df, transform=transform)
    #
    # # Defining the dataloader
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,
    #                                                batch_size=batch_size,
    #                                                shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset,
    #                                               batch_size=batch_size,
    #                                               shuffle=True)

    # if check_dataLoad:
    #     # Testing if the data has been loaded successfully into the dataloader by displaying a single picture.
    #     train_features, yield_data = next(iter(train_dataloader))
    #
    #     # Display image and label.
    #     print(f"Feature batch shape: {train_features.size()}")
    #     print(f"Labels batch shape: {yield_data.size()}")
    #
    #     img = train_features[0].squeeze()
    #     yield_local = yield_data[0]
    #     print(f"yield: {yield_local}")

    # Creating the model and loading the model to the GPU

    if model_structure == 'ResNet':
        model = Model_resnet.YieldModel().to(device)
    elif model_structure == 'EfficientNet':
        model = EfficientNet_model.YieldModel().to(device)
    else:
        model = Model_Custom.YieldModel().to(device)
    if not recover:
        model.load_state_dict(torch.load(last_save))
        init_epoc = last_epoch
    print(model)

    # Defining the optimizer to be used in the model
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'LARS':
        optimizer = LARS.LARS([params for params in model.parameters() if params.requires_grad],
                              lr=learning_rate,
                              weight_decay=1e-6,
                              exclude_from_weight_decay=["batch_normalization", "bias"],
                              )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # THe main training loop
    for t in range(init_epoc, epochs):
        i = int(t / data_change_epoch)
        if t % data_change_epoch == 0:
            train_df, test_df = X_train[i], X_test[i]
            train_df = train_df.reset_index()
            test_df = test_df.reset_index()

            # Defining the dataset from the generated Dataframe
            train_dataset = Dataloader.CustomImageDataset(annotations_file=train_df, transform=transform)
            test_dataset = Dataloader.CustomImageDataset(annotations_file=test_df, transform=transform)

            # Defining the dataloader
            train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=True)

        print(f"Epoch {t + 1}\n-------------------------------")
        print(f"Running with cross validation dataset {i + 1}")
        ep_loss, ep_accuracy, val_epoch_loss, val_epoch_accuracy, r2_train, r2_val, data_val = train_loop.train_loop(
            train_dataloader,
            test_dataloader, model,
            loss_fn, optimizer, t,
            min_max)
        if not Debug:
            writer.add_scalar('Training/ep_loss', sum(ep_loss) / len(ep_loss), t)
            writer.add_scalar('Training/ep_accuracy', sum(ep_accuracy) / len(ep_accuracy), t)
            writer.add_scalar('Evaluation/ep_loss', sum(val_epoch_loss) / len(val_epoch_loss), t)
            writer.add_scalar('Evaluation/ep_accuracy', sum(val_epoch_accuracy) / len(val_epoch_accuracy), t)
            writer.add_scalar('Training/RMSE', data_val[0], t)
            writer.add_scalar('Training/MAE', data_val[1], t)
            writer.add_scalar('Evaluation/RMSE', data_val[2], t)
            writer.add_scalar('Evaluation/MAE', data_val[3], t)
            if r2_val < 0:
                writer.add_scalar('Training/r2_score', 0.0, t)
                writer.add_scalar('Training/r2_percent', 0.0, t)
                writer.add_scalar('Evaluation/r2_score', 0.0, t)
                writer.add_scalar('Evaluation/r2_percent', 0.0, t)
            else:
                writer.add_scalar('Training/r2_score', r2_train, t)
                writer.add_scalar('Training/r2_percent', r2_train * 100, t)
                writer.add_scalar('Evaluation/r2_score', r2_val, t)
                writer.add_scalar('Evaluation/r2_percent', r2_val * 100, t)

        if t % 10 == 0:
            # save model
            PATH = f'{model_savePath}/{run_name}_runtime_save.pth'
            torch.save(model.state_dict(), PATH)
            data['last_epoch'] = t
            data['last_save'] = PATH
            data['program_complete'] = False
            with open(config_file, "w") as yamlfile:
                documents = yaml.dump(data, yamlfile)
                print("YAML updated successfully")
    print("Training Done!")
    PATH = f'{model_savePath}/{run_name}_final.pth'
    torch.save(model.state_dict(), PATH)

    data['last_epoch'] = epochs
    data['program_complete'] = True
    data['last_save'] = PATH

    with open(config_file, "w") as yamlfile:
        documents = yaml.dump(data, yamlfile)
        print("YAML updated successfully")

    if not Debug:
        writer.close()
