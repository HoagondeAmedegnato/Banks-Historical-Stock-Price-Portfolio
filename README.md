# Banks-Historical-Stock-Price-Portfolio
Visualize and Analyze the Stock Prices of 6 major Banks
___
**Note**: This readme will guide you through the steps I took to complete this project
___
**Table of Contents**
___
1. ***Introduction***

2. ***Data Collection and Data Preprocessing***
3. ***Exploratory Data Analysis***: 00.EDA.ipynb 
4. ***Feature Engineering***: 01.Assets Allocation.ipynb
5. ***Model Development***: 02.CAPM.ipynb
6. ***Inference***: 03.Prediction.ipynb
7. ***Findings and Takeaways***
___
1- ***Introduction*** 

In this project, I chose NOT only to respond to the requirements of a take home, which is to Visualize and Analyze the stock prices of the six (6) major banks, but also to go over to develop some others skillsets in Financial Analysis, such as Assets Allocation, Capital Asset Pricing Model, in order to have a better understanding of the bank industry and how it behaves.
The goal of this project, is to *Help Investors Make Important Decisions And Predict New Trends*.
___
2- ***Data Collection and Data Preprocessing***

The datasets I use, is from Kaggle website (Data Source: https://www.kaggle.com/tomasmantero/banks-historical-stock-price), where 99 csv files datasets have been provided. Each file has stock information from a specific bank or Financial Service company from Jan 1st, 2006 to Nov 1st, 2020. And from these 99 csv files, I ONLY chose the six (6) banks csv files to work on, which are:
- Bank of America (BAC)
- CitiGroup (C)
- Goldman Sachs (GS)
- JPMorgan Chase (JPM)
- Morgan Stanley (MS)
- Wells Fargo (WFC)

Each dataset has Feature Columns as follow:

- *High*: Is the highest price at which a stock traded during the course of the trading day.
- *Low*: Is the lowest price at which a stock traded during the course of the trading day.
- *Open*: Is the price at which a stock started trading when the opening bell rang.
- *Close*: Is the last price at which a stock trades during a regular trading session.
- *Volume*: Is the number of shares that changed hands during a given day.
- *Adj Close*: The adjusted closing price amends a stock's closing price to reflect that stock's value after accounting for any corporate actions. Factors in corporate actions, such as stock splits, dividends, and rights offerings.

Each dataset has the same length of 3749

I’ve downloaded these six (6) csv files from the Kaggle website and save them into my Google Drive, created a Google Colaboratory for the notebooks.

I’ve created four (4) Google Colaboratory, because my project is subdivided into four (4) parts:

- Exploratory Data Analysis (EDA), where I’ve explored the analysis for the overall datasets
- Portfolio Assets Allocation and Statistical Data Analysis, where I’ve calculated the portfolio daily return, visualized them, and calculated the portfolio statistical metrics, such as cumulative return, average daily return, and Sharpe ratio, using the selected (Close) column of the dataset.
- Capital Asset Pricing Model (CAPM), where I’ve calculated Beta for a single stock return (BAC and JPM for example) and fit a polynomial between them, applied the CAPM formula to an individual stock return, defined a function to calculate Beta for all stocks returns, applied the CAPM formula to calculate the return for the portfolio.
- Predict Banks Stocks Future Prices using Machine Learning and Deep Learning, where I’ve trained a ridge regression model and deep neural network model to predict future stock prices.

During the project, I’ve also responded to these following questions that have been asked from the take home:

- What is the max Close price for each bank's stock throughout the time period?
- On what date did Citigroup stock reach its highest price?
- Why does the first row have NaN values?
- Is there a stock that stands out?
- Did anything significant happen on 2009-01-20?
- Which stock would you classify as the riskiest over the entire time period?
- Which would you classify as the riskiest for the year 2015?
___
3- ***Exploratory Data Analysis***: 00.EDA.ipynb

As part of this, I’ve implemented the following steps:

- Analyzed the whole datasets by examining them
- Created a dataframe named Bank Stocks Close, by selecting the close column, in order to analyze the price of the bank’s stocks
- Calculated the daily Return of each Bank on the Stock’s Price; the formula I used for this return is as follow: ***Bank_daily_return[j] = (df[j]-df[j-1])/df[j-1]***; where df[j] = stock price of today; df[j-1] = stock price from previous day. So, the Return is calculated as stock price of today minus stock price from previous day, divided by stock price from previous day.
- Calculated the Pearson Correlation of the stocks close and stocks return. 
___
4- ***Feature Engineering***: 01.Assets Allocation.ipynb

Assuming I have 1 million of dollars to be invested, and I will allocate this fund based on the weights of the stocks.  I’ve defined the ASSETS = [BAC  C  GS  JPM  MS  WFC], and the WEIGHTS = [10.92%  12.06%  0.602%  3.627%  14.492%  17.636%], Portfolio weights must sum to 1: I’ve created a function (portfolio) to normalize the stock prices based on their initial price, calculated the daily return of that portfolio, created a function that takes in the stock prices along with the weights and return. I’ve calculated the Cumulative return, the Standard deviation, the Average daily return, and the Sharpe ratio of the portfolio. Note that:
- Stock daily return is a calculation of how much investors have gained or lost per day. The formula is: ***Stock daily return = Closing stock price(t) – Closing stock price(t-1) / Closing stock price(t-1)***
- Cumulative return is a measure of the aggregate amount that the stock gained or lost over a period of time. The formula is: ***Stock Cumulative return = Current price of stock – Original price of stock / Original price of stock***
- The Standard deviation is a measurement of the dispersion away from the mean. The more spread the data is, the higher the standard deviation.
- Sharpe ratio is used by investors to calculate the return of an investment compared to its risk. It’s a simply a calculation of the average return earned in excess of the risk free rate. As Sharpe ratio increases, risk-adjusted return increases and security becomes more desired by investors.
___
5- ***Model Development***: 02.CAPM.ipynb

I’ve developed CAPM (Capital Asset Pricing Model), which is a model that describes the relationship between the expected return and risk of securities. It indicated that the expected return on a security is equal to the risk-free return plus a risk premium. CAPM assumes that there exists a risk-free asset with zero standard deviation. The CAPM formula is as follow: ***r(i) = r(f) + B(i)(r(m) – r(f))***, where r(i) = Expected Return of a Security; r(f) = Risk Free Rate of Return; B(i) = Beta between the stock and the market; r(m) – r(f) = Risk Premium (incentive for investing in a risky security).

As part of this, I’ve implemented the following steps:

- Calculated Beta for a single stock return (BAC and JPM for example) and fit a polynomial between them. Note that Beta represents the slope of the line regression line (market return vs. stock return); it’s a measure of the volatility or systematic risk of a security or portfolio compared to the entire market (JPM). Beta is used in the CAPM and describes the relationship between systematic risk and expected return for assets
- Applied the CAPM formula to an individual stock return 
- Defined a function to calculate Beta for all stocks returns
- Applied the CAPM formula to calculate the return for the portfolio
___
6- ***Inference***: 03.Prediction.ipynb

In this part of prediction, I’ve trained a ridge regression model and deep neural network model to predict future stock prices. By accurately predicting stock prices, investors can maximize returns and know when to buy/sell securities. The AI/ML model will be trained using historical stock price data along with the volume of transaction. I use a type of neural nets known as Long Short-Term Memory Networks (LSTM).

As part of this, I’ve implemented the following steps:

- Created a dataframe named Bank Stocks Volume, by selecting the volume column of the datasets, in order to analyze the volume of the bank’s stocks
- Using the Bank Stocks Price and the Bank Stock Volume dataframes, I divided the datasets into 2 parts to train the AI/ML model: 65% for the training, 35% for the testing. The training set is used for model training, and the testing set is used for testing model. Below is for example the training and the testing of BAC
- I also built and trained a Ridge Linear Regression Model to test the model and to make a prediction, by choosing alpha = 1 and 4
- I've also trained a Long Short-Term Memory Network (LSTM) Time Series Model for WFC and GS, using 70% for the training and 30% for the testing, and made that prediction. Note that LSTM networks are type of RNN (Recurrent Neural Network) that are designed to remember long term dependencies by default. It can remember and recall information for a prolonged period of time. LSTM contains gates that can allow or block information from passing by.
___

7- ***Findings and Takeaways***
