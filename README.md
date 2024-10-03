# Stock Prediction
This project aims to create a working ML algorithm that predicts stock trend. 
By using Neural Network LSTM algorithm, we train the model to accurately predicts prices with technical indicators such as Moving Averages, RSI, MACD etc., and with this algorithm trained, we aim to implement this model into real trading strategies by creating a trading bot of our own.
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
### Prerequisites
- Make sure you have docker (Podman/Orbstack) installed.
- If you are using Nvidia GPU make sure you have configured GPU drivers correctly.
- For more information regarding setting up cuda toolkit with docker container, 
refer to https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
### Setting up container environment
1. Clone the repository
```bash
 git clone https://github.com/MaxymHuang/Stock_Prediction.git
```
2. Build image
navigate to the directory and build the docker image
```bash
 docker build -t stock_prediction .
```
## Usage
Run the image to access jupyter notebook
```bash
docker run -p 8888:8888 -v $(pwd):/app --gpus all stock_prediction
```
## License
This project is licensed under the [MIT License](LICENSE).
