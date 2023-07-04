# ScanVerdict-ML

## About 

This repository holds the back-end part of the ScanVerdict web application which covers the Data Science process of the application. This process extends from data acquisition through web scraping, to data cleaning and processing, model evaluation and ultimately data visualisations. 

## Installation and Requirements

This project requires Python and the following main libraries installed:

- selenium==4.10.0
- chromedriver-autoinstaller==0.4.0
- numpy==1.24.3
- pandas==2.0.2
- langdetect==1.0.9
- deep-translator==1.11.1
- transformers==4.30.2
- scikit-learn==1.3.0
- plotly==5.15.0

Or you can download all of these with this command as well:  
```bash
pip install -r requirements.txt
```

The first run will take some time, approximately 5 minutes, because some dependencies need to be installed for the word analysis. The following runs should be faster. 

## Key File Description

execute_analysis.py contains the whole data analysis workflow excluding the EDA and the evaluation of the model. 
app.py contains the api call between the front and the back. 

## Run API

To run the flask API, run in the terminal: 
```bash
Flask --app app run
```
