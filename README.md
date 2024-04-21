# SWHPredict
This repository is the implementation of significant wave height (SWH) prediction based on deep learning models. It includes abundant models and preprocessed datasets. With a command, we can easily achive the model training and testing.

## Model

The repo includes LSTM, TCN, TCN-Attention, and Transformer models. In the future, I will continue to add models.

## Dataset

The SWH dataset is provided by [National Data Buoy Center](https://www.ndbc.noaa.gov/). NDBC records ocean and meteorological data from hundreds of buoy stations around the world, providing free access for marine researchers to learn and use. However, the original dataset has missing values and is complicated, so preprocessing of the original dataset is necessary. The preprocessed dataset is stored in the/swhdata folder, and they are saved as CSV files by year and station.

Reminder: The preprocessed dataset provided by this repository is for reference only and is not responsible for accuracy.In addition, this dataset is only for learning or scientific research purposes, and commercial use is strictly prohibited.

## Usage

1. Install Pytorch and necessary dependencies:
```
pip3 install -r requirements.txt
```
2. Train and test the model easily with the following command:
```
# Predicting the wave height of station 41008 in the next 1 hour with LSTM model.
python3 run.py --model LSTM --station_id 41008 \
--seq_len 24 --pred_len 9 \
--features "WaveHeight" --years "2018" --batch_size 32 \
--learning_rate 0.01 --train_epochs 80 --activation Relu \
--data_rootpath /root/swh_dataset/

# Predicting the wave height of station 42055 in the next 12 hour with Transformer model.
python3 run.py --model Transformer --station_id 42055 \
--seq_len 24 --label_len 6 --pred_len 12 \
--e_layers 2 --d_ff 16 --enc_in 1 --dec_in 1 --c_out 1 --d_model 32 \
--features "WaveHeight" --years "2018 2022" \
--batch_size 32 --learning_rate 0.001 --train_epochs 50
--data_rootpath /root/swh_dataset/
```
