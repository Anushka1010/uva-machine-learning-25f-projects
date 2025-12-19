# BERT for Financial Sentiment Analysis

Team 24 Machine Learning Final Project

## Team Members
- Colby Wise (bpx9hy)
- Flavien Moise (nvu5gw)

## Overview 
Our project is determining the correlation between financial news headlines and stock performance. We are using BERT to determine news headline sentiment, and statistical methods to analyze the correlation. 

## How to run our code

Install the requirements (`pip install -r requirements.txt`). It is recommended to use conda to create a Python 3.10 environment for this project. 

1. First, run the data preparer
    - `python3 ./data_preparer.py`
    - set `MAX_ROWS` on line 229 to the amount of data you wish to process, or `None` to process all data, which may take a while. 
    - The location for the input file on 227 may need to change to reflect the location of the data. It will currently look for the json to be in the same folder as the code, and may need to be changed to "data/polygon_news_sample.json" if the json is in the data folder. 
2. Next, run the BERT model
    - `python3 ./bert.py`
3. Last, run the statistical analysis
    - `python3 ./statistical_analysis.py`

Note that Python is commonly installed as `py` on Windows. 

A video of the code running is available here: https://drive.google.com/file/d/1_knZ4cJ5R_VtIu8JD36ALOirAj8IVv1H/view?usp=sharing 