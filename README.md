# SWHPredict
This repository is the implementation of significant wave height (SWH) prediction based on deep learning models. It includes abundant models and preprocessed datasets. With a command, we can easily achive the model training and testing.

## Model

The repo includes LSTM, TCN, TCN-Attention, and Transformer models. In the future, I will continue to add models.

## Dataset

The SWH dataset is provided by [National Data Buoy Center]{https://www.ndbc.noaa.gov/}. NDBC records ocean and meteorological data from hundreds of buoy stations around the world, providing free access for marine researchers to learn and use. However, the original dataset has missing values and is complicated, so preprocessing of the original dataset is necessary. The preprocessed dataset is stored in the/swhdata folder, and they are saved as CSV files by year and station.

Reminder: The preprocessed dataset provided by this repository is for reference only and is not responsible for accuracy.In addition, this dataset is only for learning or scientific research purposes, and commercial use is strictly prohibited.
