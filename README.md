# Predicting US Stocks Returns Direction (UP/DOWN)
## Implementing Anchor bias theory 

[My Research Paper for the Subject!](https://www.academia.edu/40213993/Algorithmic_Trading_High_Frequency_and_Low_Frequency_Trading?source=swp_share)

### Problem Identification :
Investors are anchored to the long term moving averages. The long term moving average is defined by the 252 moving average, and the short term is defined by the 21-Day moving average. The distance between the two moving averages is the moving average distance (MAD = 21-DAY MA / 252-DAY MA). When the MAD>1, the ditance is called a positive spread and when MAD< 1, the distance is caleed a negative spread. 

The ancnchor bias theory, published in a research paper by Avramov, Kaplanski, Subrahmanyam(2018), states that when MAD spread is positive positive announcment (sentiment) drive the price of the stocks to go up more than than negative sentiment drive the price to go down. However, when MAD spread is negative, negative sentiment drives price to go down more than positive sentiment drives re price to go up. Noting that the larger/ smaller the MAD, in both cases, the more effective is the strategy

The model proposed is to predict US stocks returns ( +/-) based on several features but mainly on a BUY or SELL signal. The engineered feature, named trading signal is the main feature which is processed by the constructed pipeline. The BUY signal is construcetd by getting positive sentiment from 2 databases (Sentdex and stocktwits), a 7 days previous senitiment score and a positive MAD greater than 1.2. The SELL signal is set based on negative sentiment scores from 2 databases, also a 7 day previous negative score and a negative MAD less than 0.8.

The stated signals are passed to the pipeline to pass through more than 8000 US stocks and filter out each day, the stocks that passed the criteria. Several screens where passed to the timeline to insure no stock has a null sentiment score (in any of the two databases) or a zero return ( which was actually found).Several other features where passed to the pipeline to output a dataframe of the filtered stocks. After doing the nessary transformations, the data is based to two machine learing algorithms.
