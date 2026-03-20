---
Client Machine: [Raspberry Pi, Jetson nano, Ubuntu machine]
Server Machine: Mac mini
dataset: [MMASH, RRITSH, SmartCare]
---

---
framework: [Flower, Python library (tensorflow, keras, scikit-learn, matplotlib, pandas, numpy)]
---

# Federated AI with Embedded Devices using Flower

We have implemented using Flower documentation written for embedded devices. Github link is provided below.

https://github.com/flwrlabs/flower/tree/main/examples/embedded-devices


## Getting things ready


## Clone this example

> \[!NOTE\]
> Cloning the example and installing the project is only needed for the machine that's going to start the run. The embedded devices would typically run a Flower `SuperNode` for which only `flwr` and relevant libraries needed to run the `ClientApp` (more on this later) are needed.

Start with cloning this example on your laptop or desktop machine. We have prepared a single line which you can copy and execute:


Create a new directory called `embedded-devices` with the following structure:

```shell
embedded-devices
├── embeddedexample
│   ├── __init__.py
│   ├── client_app.py      # Defines your ClientApp
│   ├── server_app.py      # Defines your ServerApp
│   ├── task.py            # Defines your model, training and data loading
│   └── preprocess.py      # Handles data preprocessing (e.g., scaling, windowing, cleaning)
├── pyproject.toml         # Project metadata like dependencies and configs
└── README.md
```

Install the dependencies defined in `pyproject.toml`.

We initially implemented two client application both with Jetson nano and later on we used three heterogeneous clients Raspberry Pi, Jetson nano and Ubuntu machine. For each DL models we created separate subfolders. Copy and paste the subfolders for analysis on each model every time if you want to implement this. The structure should be like below 

```shell
embedded-devices
├── embeddedexample
│   ├── __init__.py
│   ├── client_app.py      # Defines your ClientApp
│   ├── server_app.py      # Defines your ServerApp
│   ├── task.py            # Defines your model, training and data loading
│   └── preprocess.py      # Handles data preprocessing (e.g., scaling, windowing, cleaning)
├── pyproject.toml         # Project metadata like dependencies and configs
└── README.md
```


### Launching the Flower `SuperLink`

On your development machine, launch the `SuperLink`. You will connnect Flower `SuperNodes` to it in the next step.

> \[!NOTE\]
> If you decide to run the `SuperLink` in a different machine, you'll need to adjust the `address` under the `[tool.flwr.federations.embedded-federation]` tag in the `pyproject.toml`.

```shell
flower-superlink --insecure
```

### Connecting Flower `SuperNodes`

With the `SuperLink` up and running, now let's launch a `SuperNode` on each embedded device. In order to do this ensure you know what the IP of the machine running the `SuperLink` is and that you have copied the data to the device. Note with `--node-config` we set a key named `dataset-path`. That's the one expected by the `client_fn()` in [client_app.py](embeddedexample/client_app.py). This file will be automatically delivered to the `SuperNode` so it knows how to execute the `ClientApp` logic.

Each client device should have datasets stored in it. Please refer to our paper for detail explanation.

Now, launch your `SuperNode` pointing it to the dataset you `scp`-ed earlier:

```shell
# Repeat for each embedded device (adjust SuperLink IP and dataset-path)
flower-supernode --insecure --superlink="SUPERLINK_IP:9092" \
                 --node-config="dataset-path='path/to/dataset'"
```

Repeat for each embedded device that you want to connect to the `SuperLink`.

### Run the Flower App

With both the long-running server (`SuperLink`) and two `SuperNodes` up and running, we can now start run. Note that the command below points to a federation named `embedded-federation`. Its entry point is defined in the `pyproject.toml`. Run the following from your development machine where you have cloned this example to, e.g. your laptop.

```shell
flwr run . embedded-federation
```
