# # import torch
# # from ray import train, tune
# # from ray.tune.search.optuna import OptunaSearch


# # def objective(config):  # ①
# #     train_loader, test_loader = load_data()  # Load some data
# #     model = ConvNet().to("cpu")  # Create a PyTorch conv net
# #     optimizer = torch.optim.SGD(  # Tune the optimizer
# #         model.parameters(), lr=config["lr"], momentum=config["momentum"]
# #     )

# #     while True:
# #         train_epoch(model, optimizer, train_loader)  # Train the model
# #         acc = test(model, test_loader)  # Compute test accuracy
# #         train.report({"mean_accuracy": acc})  # Report to Tune


# # search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
# # algo = OptunaSearch()  # ②

# # tuner = tune.Tuner(  # ③
# #     objective,
# #     tune_config=tune.TuneConfig(
# #         metric="mean_accuracy",
# #         mode="max",
# #         search_alg=algo,
# #     ),
# #     run_config=train.RunConfig(
# #         stop={"training_iteration": 5},
# #     ),
# #     param_space=search_space,
# # )
# # results = tuner.fit()
# # print("Best config is:", results.get_best_result().config)





# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from ray import train, tune
# from ray.tune.search.optuna import OptunaSearch

# # Example data generation
# def load_data():
#     # Generate some dummy data for illustration purposes
#     X_train = np.random.rand(100, 1, 28, 28).astype(np.float32)
#     y_train = np.random.randint(0, 10, 100).astype(np.int64)
#     X_test = np.random.rand(20, 1, 28, 28).astype(np.float32)
#     y_test = np.random.randint(0, 10, 20).astype(np.int64)
    
#     train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
#     return train_loader, test_loader

# # Example convolutional neural network
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(32 * 14 * 14, 10)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = x.view(-1, 32 * 14 * 14)
#         x = self.fc1(x)
#         return x

# # Training function
# def train_epoch(model, optimizer, train_loader):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

# # Testing function
# def test(model, test_loader):
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     return correct / len(test_loader.dataset)

# def objective(config):
#     train_loader, test_loader = load_data()
#     model = ConvNet().to("cpu")
#     optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    
#     for _ in range(5):  # Number of training iterations
#         train_epoch(model, optimizer, train_loader)
#         acc = test(model, test_loader)
#         train.report({"mean_accuracy": acc})

# search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
# algo = OptunaSearch()

# tuner = tune.Tuner(
#     objective,
#     tune_config=tune.TuneConfig(
#         metric="mean_accuracy",
#         mode="max",
#         search_alg=algo,
#     ),
#     run_config=train.RunConfig(
#         stop={"training_iteration": 5},
#     ),
#     param_space=search_space,
# )

# results = tuner.fit()
# print("Best config is:", results.get_best_result().config)


import argparse
import os

from filelock import FileLock
from tensorflow.keras.datasets import mnist

import ray
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.keras import ReportCheckpointCallback


def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf

    batch_size = 128
    num_classes = 10
    epochs = 12

    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(config["hidden"], activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(lr=config["lr"], momentum=config["momentum"]),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[ReportCheckpointCallback(metrics={"mean_accuracy": "accuracy"})],
    )


def tune_mnist():
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources={"cpu": 2, "gpu": 0}),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=10,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={"mean_accuracy": 0.99},
        ),
        param_space={
            "threads": 2,
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
            "hidden": tune.randint(32, 512),
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

tune_mnist()