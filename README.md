# Predicting Internet Traffic on Amazon

## Repository Contents

This repository contains code, data and figures for a data science project investigating and forecasting internet traffic on Amazon, one of the most visited websites in the world. This project was created as part of the Data Science Project Course (DS 4002) at the University of Virginia in the Fall of 2022.

## Source Code

### Installing/Building Code in this Repository

After cloning or forking this repository, its contents can be used to recreate different parts of this project. The required modules used in Python this project are listed below.

### Modules Used in this Project

#### Python Modules

This project makes use of the following Python modules:
- `datetime` - to convert date values to datetime objects
- `matplotlib.pyplot` - to create plots
- `numpy` - for numerical data manipulation
- `os` - for setting the working directory
- `pandas` - for data manipulation
- `pmdarima` - to create ARIMA model
- `sklearn` - for preprocessing and support vector classifier
- `statsmodels.tsa.api` - to train the model
- `statsmodels.api` - to test stationarity

### Usage of Code in this Repository

There is one code file in this repository which can be found in the `src` directory of this repository: `TimeSeriesModeling.py` ([src](src/TimeSeriesModeling.py)). This file contains the code for visualization of the time series data, as well as the creation and evaluation of the machine learning forecasting model. The data is read in and then filtered to include only our Amazon data. The data us then initially visualized to detect the trends we are interested in. Then we pre-process the data for modeling, by detecing outliers, testing for stationarity and seasonality. This section of the code was based on a time series analysis blog post found on the "Towards Data Science" site [3]. After this, we split the data into testing and training. Following this, we fit and train the model. We then create the forecast and evaluate it on the test data. Lastly, we find the relevant metrics for this analysis. The model creation and evaluations was based on the code found a blog post on "Medium" site [4].

## Data

The dataset to be used for this project contains page visit data for different websites over time. The data was collected from Semrush, a website that provides website traffic information for different domains. Data for our different websites of interest were downloaded on Semrush, and then all compiled in Excel. The dataset contains page visits for each month from January 2017 to October 2022 for our 8 separate websites, so there are 560 total records. After we decided Amazon would be our only website of interest, we filtered it down for just Amazon, which left us with 70 observations. The full data file can be found in CSV format in the `data` directory of this repository in `siteVisitsByMonth.csv` ([src](data/siteVisitsByMonth.csv)).

### Data Dictionary

| Variable | Data Type | Description | Example |
|----------|-----------|-------------|---------|
| Site | String | Whe website of interest | 'Amazon' |
| MonthYear | Date | The month and year corresponding to the page views | 2017-01-01 |
| Year | Numeric | The year the website was visited | 2017 |
| AllDevices | Numeric | The number of page devices for the month and year across all devices | 2571858 |
| Desktop | Numeric | The number of page devices for the month and year across desktop devices | 6565 |
| Mobile | Numeric | The number of page devices for the month and year across mobile devices | 233 |
| Category | String | The category/industry that the website falls into. One of: 'E-commerce', 'News', 'Sports' | 'E-commerce'

## Figures

Figures for this project can be found in the `figures` directory of this repository.

### Table of Contents

| Figure Name | Variables | Summary |
|-------------|-----------|---------|
| Amazon Page Visits Over Time | x = date, y = page visits (in billions) | The line graph shows the trend and moving average of page visits on Amazon since 2017. The overall trend we see is that page visits are increasing. There also appears to roughly be patterns of seasonality, with higher page visits towards the ends and beginnings of the year, with lower numbers in the middle of the year. |
| Outlier Detection | x = date, y = page visits | This plot shows the same line graph as the prior graph, but points out the outliers that were calculated in the time series. From this, we know May 2017, September 2017, August 2022, and October 2022 are all outliers to be removed for further analysis. |
| SARIMA (1,1,1) x (1,1,1,12) | x and y are different variables relevant to each plot | The series of plot found in this figure summarize the time series forecasting model and its evaluated metrics. From the "Forecast" model, we see our predictions are working relatively well, following the general pattern from the observed data. From the "Residuals" plot, we see that the residuals appear to be stationary with no changing patterns, which is a good sign. From the "Residual Distribution" plot, we see the residuals are roughly normally distributed, which is appropriate. |
| Future Forecast | x = date, y = page visits | The graph displays the future forecasted value until the end of 2024, with the model now fitted to the entire dataset. We see the expected increasing trend, with occassional dips in the early/middle parts of the year, and spikes towards the end of the years. |

## References

[1] R. Jogi, “How to Handle Heavy Internet Traffic on Your Website?,” Cloud Minister Technologies. Sept. 28, 2021. [Online]. Available: https://cloudminister.com. [Accessed Oct. 5, 2022].

[2] A. Coghlan, “Using R for Time Series Analysis,” Little Book of R for Time Series. 2010. [Online]. Available: https://a-little-book-of-r-for-time-series.readthedocs.io. [Accessed Oct. 12, 2022].

[3] M. Pietro, “Time Series Analysis for Machine Learning,” Towards Data Science. 2020. [Online]. Available: https://towardsdatascience.com. [Accessed Oct. 17, 2022].

[4] M. Pietro, “Time Series Forecasting: ARIMA vs LSTM vs PROPHET,” Medium. 2020. [Online]. Available: https://medium.com. [Accessed Oct. 17, 2022].

Files documenting the previous 2 milestones of this project can be found in the `milestones` directory of this repository in `M1Hypothesis.pdf` ([src](milestones/MI1Hypothesis.pdf)) and `MI2EstablishDataAndAnalysisPlan.pdf` ([src](milestones/MI2EstablishDataAndAnalysisPlan.pdf)).
