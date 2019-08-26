#!/usr/bin/env python
# coding: utf-8

# Notes:
# df.ix[:] : returns sliced rows based on passed indexes

# # Predicting US Stocks Returns Direction (UP/DOWN)
# ## Implementing Anchor bias theory 
# 
# 
# ### Problem Identification :
# Investors are anchored to the long term moving averages. The long term moving average is defined by the 252 moving average, and the short term is defined by the 21-Day moving average. The distance between the two moving averages is the moving average distance (MAD = 21-DAY MA / 252-DAY MA). When the MAD>1, the ditance is called a positive spread and when MAD< 1, the distance is caleed a negative spread. 
# 
# The ancnchor bias theory, published in a research paper by Avramov, Kaplanski, Subrahmanyam(2018), states that when MAD spread is positive positive announcment (sentiment) drive the price of the stocks to go up more than than negative sentiment drive the price to go down. However, when MAD spread is negative, negative sentiment drives price to go down more than positive sentiment drives re price to go up. Noting that the larger/ smaller the MAD, in both cases, the more effective is the strategy
# 
# The model proposed is to predict US stocks returns ( +/-) based on several features but mainly on a BUY or SELL signal. The engineered feature, named trading signal is the main feature which is processed by the constructed pipeline. The BUY signal is construcetd by getting positive sentiment from 2 databases (Sentdex and stocktwits), a 7 days previous senitiment score and a positive MAD greater than 1.2. The SELL signal is set based on negative sentiment scores from 2 databases, also a 7 day previous negative score and a negative MAD less than 0.8.
# 
# The stated signals are passed to the pipeline to pass through more than 8000 US stocks and filter out each day, the stocks that passed the criteria. Several screens where passed to the timeline to insure no stock has a null sentiment score (in any of the two databases) or a zero return ( which was actually found).Several other features where passed to the pipeline to output a dataframe of the filtered stocks. After doing the nessary transformations, the data is based to two machine learing algorithms.

# ## Data Gathering using a Pipeline:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# In[2]:


# Import Pipeline class and datasets
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import USEquityPricing
from quantopian.pipeline.data.psychsignal import stocktwits
# from quantopian.interactive.data.sentdex import sentiment
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.factors import CustomFactor, MarketCap, Latest
from quantopian.pipeline.classifiers.fundamentals import SuperSector
# Import built-in moving average calculation
from quantopian.pipeline.factors import SimpleMovingAverage, DailyReturns, Returns

# Import built-in trading universe
from quantopian.pipeline.experimental import QTradableStocksUS

# Define our own custom factor class
class SentimentSevenDaysAgo(CustomFactor):
    inputs = [sentiment.sentiment_signal]
    window_length=7

    def compute(self, today, assets, out, sentiment):
        out[:] = sentiment[0] #When I specified a window _length of 7 it gave back last 7 scores thus call the senitment Boundcolumn from the top index


def make_pipeline():
    # Create a reference to our trading universe
    base_universe = QTradableStocksUS()

    # Get latest closing price
    close_price = USEquityPricing.close.latest
    
    daily_returns = DailyReturns(
        inputs = [USEquityPricing.close])
    
    returns_3 = Returns(
        inputs = [USEquityPricing.close],
        window_length = 2)
    
    
    mean_close_21 = SimpleMovingAverage(
        inputs = [USEquityPricing.close],
        window_length= 21)
    
    mean_close_252 = SimpleMovingAverage(
        inputs = [USEquityPricing.close],
        window_length= 252)    
    
    MAD = mean_close_21 / mean_close_252
    
    sentdex_score = sentiment.sentiment_signal.latest
    
    sentdex_lag = SentimentSevenDaysAgo()
     
    marketCap = MarketCap()
    
    stock_class = SuperSector()
    
    not_zero_returns = (daily_returns != 0) & (returns_3 !=0)
    # Calculate 3 day average of bull_minus_bear scores
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=2,
    )
    # Create filter for positive/negative spread moving averages
    # Create filter for positive sentiment scores
    # Crate filter for 7 days lag sentiment score
    # assets based on their sentiment scores
    positive_MAD = MAD > 1.2
    negative_MAD = MAD < 0.8
    
    positive_sentiment_lag = sentdex_lag > 3
    negative_sentiment_lag = sentdex_lag < -1
    
    positive_sentiment = sentdex_score > 2
    negative_sentiment = sentdex_score < -1
    
    positive_twits = sentiment_score > 0
    negative_twits = sentiment_score < 0
#     Long = sentdex_lag.top(10, mask = positive_MAD)
#     short = sentdex_lag.bottom(10, mask = negative_MAD) 

    
    

    Long = (positive_MAD & positive_sentiment_lag & positive_twits & positive_sentiment)
    short = (negative_MAD & negative_sentiment_lag & negative_twits )
    
    tradeable_equities = (Long | short)
    # Return Pipeline containing all below columns and
    # sentiment_score that has our trading universe as screen
    return Pipeline(
        columns={
            'close_price': close_price,
            "Sentdex": sentdex_score,
            "Sentdex_lag": sentdex_lag,
            'sentiment_score': sentiment_score.zscore(), #apply zscore to normalize
            "MAD": MAD,
            "BUY": Long,
            "SHORT": short,
            "return": daily_returns,
            "Returns": returns_3,
            "Market Capital.": marketCap,
            "Stock Classfiaction": stock_class
        },
        screen=(base_universe
        & tradeable_equities& sentdex_lag.notnull() & sentiment_score.notnull() & not_zero_returns)
    )


# In[3]:


# Import run_pipeline method
from quantopian.research import run_pipeline

# Specify a time range to evaluate
period_start = '2013-01-01 07:12:03.6'
period_end = '2019-01-01 07:12:03.6 '

# Execute pipeline created by make_pipeline
# between start_date and end_date
pipeline_output = run_pipeline(
    make_pipeline(),
    start_date=period_start, 
    end_date=period_end
)
# pipeline_output.add(sentiment_free.sentiment_signal, 'sentiment_signal')

# Display last 10 rows
pipeline_output.tail(20)
# print('Number of securities that passed the filter: %d' % len(pipeline_output))


# In[4]:


pipeline_output.describe()


# ## Pipeline Schema to Fetch US Tradeable Equities

# In[5]:


# pipeline_output['return'] = pd.get_dummies(pipeline_output['return'],drop_first=True)
# pipeline_output.head(30)
make_pipeline().show_graph(format='jpeg') #or png


# ## Feature Engineering

# In[ ]:





# In[4]:


# return_encoding = pd.get_dummies(pipeline_output['return'],drop_first=True)
# pipeline_output['return'] = return_encoding
# pipeline_output['Returns'] = pipeline_output["Returns"].apply(np.sign)

pipeline_output['return'] = pipeline_output["return"].apply(np.sign)
pipeline_output['Market Capital.'] = pipeline_output["Market Capital."].apply(np.log)
# Applied zscore to Market Capitalization but got an outlier which was Apple stock


# In[5]:


pipeline_output.head(10)


# In[6]:


pipeline_output["Trading Signal"] = pd.get_dummies(pipeline_output['BUY'],drop_first=True)

pipeline_output.tail(20)
# if BUY "FALSE" --> 0 | if BUY "TRUE" --> 1


# In[14]:


sns.pairplot(pipeline_output,hue='return')


# ## Data Analysis and Insights Generation

# In[9]:


n=float(len(pipeline_output[pipeline_output["return"]>0]))
m=float(len(pipeline_output[pipeline_output["Trading Signal"]==1]))
a=float(len(pipeline_output[pipeline_output["return"]<0]))
b=float(len(pipeline_output[pipeline_output["Trading Signal"]==0]))
z=float(len(pipeline_output))
print"The percentage of positive returns is:", ((n/z)*100),"%"
print"The percentage of BUY Trading Signal is:", ((m/z)*100),"%"
print"The percentage of negative returns is:", ((a/z)*100),"%"
print"The percentage of SELL Trading Signal is:", ((b/z)*100),"%"


# In[10]:


print(pipeline_output['Trading Signal'].value_counts())
print(pipeline_output['return'].value_counts())

#Unbalanced labels and datasets


# In[11]:


pipeline_output.hist();


# In[12]:


sns.boxplot(x='Trading Signal',y='Returns',data=pipeline_output,palette='rainbow', width= 0.3);
# For which I assigned buy (1), what where their returns
# For which I assigned sell(0), what where their returns


# In[13]:


plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Trading Signal']==1]["Returns"].hist(alpha=0.5,color='blue',
                                              bins=50,label='Trading Signal=1')
pipeline_output[pipeline_output['Trading Signal']==0]['Returns'].hist(alpha=0.5,color='red',
                                              bins=50,label='Trading Signal=0')
plt.legend()
plt.xlabel('Returns');

plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Returns']>0]["Trading Signal"].hist(alpha=0.5,color='blue',
                                              bins=50,label='Positive Returns')
pipeline_output[pipeline_output['Returns']<0]['Trading Signal'].hist(alpha=0.5,color='red',
                                              bins=50,label='Negative Returns')
plt.legend()
plt.xlabel('Trading Signal');


# In[14]:


plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Returns']>0]['Stock Classfiaction'].hist(alpha=0.5,color='blue',
                                              bins=50,label='Positive return')
pipeline_output[pipeline_output['Returns']<0]['Stock Classfiaction'].hist(alpha=0.5,color='red',
                                              bins=50,label='Negative Return')
plt.legend()
plt.xlabel('Stock Classification');

plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Trading Signal']==1]['Stock Classfiaction'].hist(alpha=0.5,color='blue',
                                              bins=50,label='Trading Signal=1')
pipeline_output[pipeline_output['Trading Signal']==0]['Stock Classfiaction'].hist(alpha=0.5,color='red',
                                              bins=50,label='Trading Signal=0')
plt.legend()
plt.xlabel('Stock Classification');


# In[15]:


plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Returns']>0]['Sentdex'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Positive Returns')
pipeline_output[pipeline_output['Returns']<0]['Sentdex'].hist(alpha=0.5,color='red',
                                              bins=30,label='Negative Returns')
plt.legend()
plt.xlabel('Sentdex');

plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Returns']>0]['Sentdex_lag'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Positive Returns')
pipeline_output[pipeline_output['Returns']<0]['Sentdex_lag'].hist(alpha=0.5,color='red',
                                              bins=30,label='Negative Returns')
plt.legend()
plt.xlabel('Sentdex 7-Day Lag');

plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Returns']>0]['sentiment_score'].hist(alpha=0.5,color='blue',
                                              bins=50,label='Positive Returns')
pipeline_output[pipeline_output['Returns']<0]['sentiment_score'].hist(alpha=0.5,color='red',
                                              bins=50,label='Negative Returns')
plt.legend()
plt.xlabel('Stocktwits Sentiment');


# In[16]:


plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Trading Signal']==1]['Market Capital.'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Positive Returns')
pipeline_output[pipeline_output['Trading Signal']==0]['Market Capital.'].hist(alpha=0.5,color='red',
                                              bins=30,label='Negative Returns')
plt.legend()
plt.xlabel('Market Capitalization');

plt.figure(figsize=(10,6))
pipeline_output[pipeline_output['Returns']>0]['Market Capital.'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Positive Returns')
pipeline_output[pipeline_output['Returns']<0]['Market Capital.'].hist(alpha=0.5,color='red',
                                              bins=30,label='Negative Returns')
plt.legend()
plt.xlabel('Market Capitalization');


# In[17]:


print('Number of securities that passed the filter: %d' % len(pipeline_output.index.levels[1].unique()))
pipeline_output.columns


# In[18]:


pipeline_output.info()


# #                          ********BUILDING THE MODEL PHASE********
# 

# In[7]:


from sklearn.linear_model import LogisticRegression
# Testing how the model's intuition 
model = LogisticRegression()
model.fit([[-2,-3],[1,0],[1,1]],["T","F","T"])
# model.coef_, model.intercept_,model.predict([-3,21])


# In[8]:


feature_cols = ["Market Capital.", "sentiment_score", "Sentdex_lag","MAD","Trading Signal", "Stock Classfiaction", "Sentdex"]
X = pipeline_output[feature_cols]
y = pipeline_output["return"] #target column (classified T or F)


# In[9]:


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state= None)


# In[10]:


# instantiate the model (using the default parameters)
logmodel = LogisticRegression()
# fit the model with data
logmodel.fit(X_train,y_train)


# In[11]:


#
y_pred=logmodel.predict(X_test)
y_pred
logmodel.coef_, logmodel.intercept_,logmodel.predict(X_test)


# In[12]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[13]:


print"Accuracy:",metrics.accuracy_score(y_test, y_pred)
print"Precision:",metrics.precision_score(y_test, y_pred)
print"Recall:",metrics.recall_score(y_test, y_pred)


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ## Determine Threshold

# In[23]:



from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score

THRESHOLD = 0.50
preds = np.where(logmodel.predict_proba(X_test)[:,1] > THRESHOLD, 1, -1)

pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                   precision_score(y_test, preds), roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"], columns = ["Scores"])
# 0.5 remained the best threshold


# In[24]:


class_names=[-1,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label');


# AUROC represents the likelihood of the model distinguishing observations from two classes.
# In other words, if a random selection of an observation from each class is made, what's the probability that your model will be able to "rank" them correctly?

# In[25]:


y_pred_proba = logmodel.predict_proba(X_test)[::,1]
fpr, tpr, threshold = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
x = np.linspace(0, 1, 100000)
plt.plot(x, x + 0, linestyle='solid',c = 'k')
plt.legend(loc=4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# ### Random Forest Classifier

# In[15]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=800, max_leaf_nodes=2,max_features=7, min_samples_split=2)
rfc.fit(X_train, y_train)


# In[16]:


from sklearn import metrics
rfc_pred = rfc.predict(X_test)
rfc_pred
cnf_matrix = metrics.confusion_matrix(y_test, rfc_pred)
cnf_matrix


# In[17]:


print("Accuracy:",metrics.accuracy_score(y_test, rfc_pred))
print("Precision:",metrics.precision_score(y_test, rfc_pred))
print("Recall:",metrics.recall_score(y_test, rfc_pred))


# In[18]:


print(classification_report(y_test,rfc_pred))


# In[41]:


y_pred_proba = rfc.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
x = np.linspace(0, 1, 100000)
plt.plot(x, x + 0, linestyle='solid',c = 'k')
plt.legend(loc=4)
plt.show()


# ## Support Vector Machine (SVMs)

# In[17]:


from sklearn.svm import SVC
sv_model = SVC(C=1,gamma=0.1)
sv_model.fit(X_train,y_train)


# In[22]:


sv_predictions = sv_model.predict(X_test)
print(metrics.confusion_matrix(y_test,sv_predictions))
print("\n")
print(classification_report(y_test,sv_predictions))


# In[22]:


#### GRIDSEARCH #### 
#Hypertuning parameters
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]} 
from sklearn.grid_search import GridSearchCV


# In[23]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)


# In[24]:


grid.fit(X_train,y_train)


# In[25]:


print(grid.best_params_)
grid.best_estimator_


# In[26]:


grid_predictions = grid.predict(X_test)
print(metrics.confusion_matrix(y_test,grid_predictions))
print("\n")
print(classification_report(y_test,grid_predictions))


# ## K-Nearest Neighbor
# ##### Run it after Pipeliene directly

# In[51]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[52]:


scaler.fit(pipeline_output.drop(['close_price', 'BUY', 'return', 'Returns', 'SHORT'],axis=1))


# In[53]:


scaled_features = scaler.transform(pipeline_output.drop(['close_price', 'BUY', 'return', 'Returns', 'SHORT'],axis=1))


# In[54]:


scaled_features


# In[56]:


pipeline_scaled = pd.DataFrame(scaled_features, columns=pipeline_output.columns.drop(['close_price', 'BUY', 'return', 'Returns', 'SHORT']))


# In[57]:


pipeline_scaled.head()


# In[49]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,pipeline_output['return'],
                                                    test_size=0.30)
# Get dummies for returns first before running


# In[70]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=36)


# In[71]:


knn.fit(X_train,y_train)


# In[72]:


pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print("\n")
print(classification_report(y_test,pred))


# In[62]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[63]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ### Out of Sample Predictions:

# In[74]:


pipeline_outsample = run_pipeline(
    make_pipeline(),
    start_date="2019-01-01", 
    end_date="2019-06-21"
)
pipeline_outsample.tail(20)


# In[75]:


pipeline_outsample['return'] = pipeline_outsample["return"].apply(np.sign)


# In[76]:


pipeline_outsample["Trading Signal"]  = pd.get_dummies(pipeline_outsample['BUY'],drop_first=True)


# In[77]:


pipeline_outsample['Market Capital.'] = pipeline_outsample["Market Capital."].apply(np.log)

pipeline_outsample.head(20)


# In[78]:


X_ofs17 = pipeline_outsample[feature_cols]
y_ofs17 = pipeline_outsample["return"]
len(X_ofs17)


# #### Fitting Logestic Model to out of sample data:

# In[79]:


y_pred_ofs17 = logmodel.predict(X_ofs17)
y_pred_ofs17


# In[80]:


cnf_matrix = metrics.confusion_matrix(y_ofs17, y_pred_ofs17)
print cnf_matrix    
print"Accuracy:",metrics.accuracy_score(y_ofs17, y_pred_ofs17)
print"Precision:",metrics.precision_score(y_ofs17, y_pred_ofs17)
print"Recall:",metrics.recall_score(y_ofs17, y_pred_ofs17)
print classification_report(y_ofs17,y_pred_ofs17)


# In[ ]:





# #### Fitting Random Forest Classifier to out of sample data:

# In[81]:


y_rfc_pred_ofs_17 = rfc.predict(X_ofs17)
y_rfc_pred_ofs_17


# In[82]:


cnf_matrix_rfc = metrics.confusion_matrix(y_ofs17, y_rfc_pred_ofs_17)
print(cnf_matrix_rfc)
print("Accuracy:",metrics.accuracy_score(y_ofs17, y_rfc_pred_ofs_17))
print("Precision:",metrics.precision_score(y_ofs17, y_rfc_pred_ofs_17))
print("Recall:",metrics.recall_score(y_ofs17, y_rfc_pred_ofs_17))
print classification_report(y_ofs17,y_rfc_pred_ofs_17)


# In[ ]:


### Fitting Support Vector Machines


# In[84]:


y_svm_pred_ofs_17 = sv_model.predict(X_ofs17)
y_svm_pred_ofs_17


# In[86]:


cnf_matrix_svm = metrics.confusion_matrix(y_ofs17, y_svm_pred_ofs_17)
print(cnf_matrix_svm)
print("Accuracy:",metrics.accuracy_score(y_ofs17, y_svm_pred_ofs_17))
print("Precision:",metrics.precision_score(y_ofs17, y_svm_pred_ofs_17))
print("Recall:",metrics.recall_score(y_ofs17, y_svm_pred_ofs_17))
print classification_report(y_ofs17,y_svm_pred_ofs_17)


# In[87]:


pipeline_outsample.tail(20)


# In[124]:


pipeline_outsample.iloc[pipeline_outsample.index.levels[0] == '2019-06-21 00:00:00+00:00']



# # Stepping Up a year and refitting both models:

# In[44]:


pipeline_output_2 = run_pipeline(
    make_pipeline(),
    start_date="2014-01-01", 
    end_date="2018-01-01"
)


# In[45]:


pipeline_output_2['return'] = pipeline_output_2["return"].apply(np.sign)
pipeline_output_2['Market Capital.'] = pipeline_output_2["Market Capital."].apply(np.log)
pipeline_output_2["Trading Signal"] = pd.get_dummies(pipeline_output_2['BUY'],drop_first=True)
 


# In[46]:


n=float(len(pipeline_output_2[pipeline_output_2["return"]>0]))
m=float(len(pipeline_output_2[pipeline_output_2["Trading Signal"]==1]))
a=float(len(pipeline_output_2[pipeline_output_2["return"]<0]))
b=float(len(pipeline_output_2[pipeline_output_2["Trading Signal"]==0]))
x=float(len(pipeline_output_2[pipeline_output_2["Returns"]>0]))
y=float(len(pipeline_output_2[pipeline_output_2["Returns"]<0]))
z=float(len(pipeline_output))
print"The percentage of positive returns is:", ((n/z)*100),"%"
print"The percentage of BUY Trading Signal is:", ((m/z)*100),"%"
print"The percentage of negative returns is:", ((a/z)*100),"%"
print"The percentage of SELL Trading Signal is:", ((b/z)*100),"%"


# In[47]:


pipeline_output_2.tail(20)


# In[48]:


X_1 = pipeline_output_2[feature_cols]
y_1 = pipeline_output_2["return"]


# In[49]:


X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(X_1,y_1,test_size=0.2,random_state= None)


# #### Fit New Logestic Model

# In[50]:


# instantiate the model (using the default parameters)
logmodel_2 = LogisticRegression()
# fit the model with data
logmodel_2.fit(X_train_1,y_train_1)


# In[51]:


y_pred_1=logmodel.predict(X_test_1)
y_pred_1


# In[52]:


cnf_matrix_1 = metrics.confusion_matrix(y_test_1, y_pred_1)
print(cnf_matrix_1)
print"Accuracy:",metrics.accuracy_score(y_test_1, y_pred_1)
print"Precision:",metrics.precision_score(y_test_1, y_pred_1)
print"Recall:",metrics.recall_score(y_test_1, y_pred_1)
print classification_report(y_test_1,y_pred_1)


# ### Fit New Random Forest Classifier Model

# In[53]:


rfc_2 = RandomForestClassifier(n_estimators=100, max_leaf_nodes=2,max_features=7, min_samples_split=1)
rfc_2.fit(X_train_1, y_train_1)


# In[54]:


rfc_pred_2 = rfc.predict(X_test_1)
rfc_pred_2
cnf_matrix_1 = metrics.confusion_matrix(y_test_1, rfc_pred_2)
print cnf_matrix_1
print "Accuracy:",metrics.accuracy_score(y_test_1, rfc_pred_2)
print "Precision:",metrics.precision_score(y_test_1, rfc_pred_2)
print "Recall:",metrics.recall_score(y_test_1, rfc_pred_2)
print classification_report(y_test_1,rfc_pred_2)


# ###  Out of Sample Predictions:

# In[55]:


pipeline_outsample_2 = run_pipeline(
    make_pipeline(),
    start_date="2018-01-01", 
    end_date="2019-01-01"
)


# In[56]:


pipeline_outsample_2['return'] = pipeline_outsample_2["return"].apply(np.sign)
pipeline_outsample_2['Market Capital.'] = pipeline_outsample_2["Market Capital."].apply(np.log)

pipeline_outsample_2["Trading Signal"] = pd.get_dummies(pipeline_outsample_2['BUY'],drop_first=True)


# In[57]:


pipeline_outsample_2.head(20)


# In[58]:


X_ofs18 = pipeline_outsample_2[feature_cols]
y_ofs18 = pipeline_outsample_2["return"]


# In[59]:


y_pred_ofs18 = logmodel_2.predict(X_ofs18)
y_pred_ofs18


# ### Logestic Regression

# In[60]:


cnf_matrix_ofs2 = metrics.confusion_matrix(y_ofs18, y_pred_ofs18)
print cnf_matrix_ofs2
print "Accuracy:",metrics.accuracy_score(y_ofs18, y_pred_ofs18)
print "Precision:",metrics.precision_score(y_ofs18, y_pred_ofs18)
print "Recall:",metrics.recall_score(y_ofs18, y_pred_ofs18)
print classification_report(y_ofs18,y_pred_ofs18)


# ### Random Forest Classifier

# In[61]:


y_rfc_pred_ofs18 = rfc_2.predict(X_ofs18)
cnf_matrix_ofs2 = metrics.confusion_matrix(y_ofs18, y_rfc_pred_ofs18)
print cnf_matrix_ofs2
print "Accuracy:",metrics.accuracy_score(y_ofs18, y_rfc_pred_ofs18)
print "Precision:",metrics.precision_score(y_ofs18, y_rfc_pred_ofs18)
print "Recall:",metrics.recall_score(y_ofs18, y_rfc_pred_ofs18)
print classification_report(y_ofs18,y_rfc_pred_ofs18)


# In[62]:


print('Number of securities that passed the filter: %d' % len(pipeline_output.index.levels[1].unique()))
print('Number of securities that passed the filter: %d' % len(pipeline_output_2.index.levels[1].unique()))
print('Number of securities that passed the filter: %d' % len(pipeline_outsample.index.levels[1].unique()))
print('Number of securities that passed the filter: %d' % len(pipeline_outsample_2.index.levels[1].unique()))


# # Use Models to Predict EOD Return's Direction:
# ## Date: 05/06/2019
# ### Wait till 7am ET for sentiment datasets to be updated
# #### PsychSignal Trader Mood : Update Frequency: Daily (updated every morning at ~7am ET)
# #### Sentdex Sentiment : Update Frequency: Daily (updated every morning at ~7am ET)
# #### US Equities Pricing : Update Frequency: Daily (updated overnight after each trading day).

# In[18]:


# Prices Update Frequency: Daily (updated overnight after each trading day).

predict = run_pipeline(
    make_pipeline(),
    start_date="2019-07-22", 
    end_date="2019-07-22"
)


# In[19]:


predict


# In[20]:


predict['Market Capital.'] = predict["Market Capital."].apply(np.log)

predict["Trading Signal"] = pd.get_dummies(predict['BUY'],drop_first=True)

predict


# In[ ]:





# In[21]:


X_live = predict[feature_cols]
print logmodel.predict(X_live)
# print logmodel_2.predict(X_live)
print rfc.predict(X_live)
# print rfc_2.predict(X_live)
print logmodel.predict_proba(X_live)
# print logmodel_2.predict_proba(X_live)
print rfc.predict_proba(X_live)
# print rfc_2.predict_proba(X_live)
print sv_model.predict(X_live)


# In[22]:


predict.index.levels[1]


# In[23]:


predict.index.levels[1]
predict['Return Predictions LR'] = logmodel.predict(X_live)
predict["Return Predictions RFC"] = rfc.predict(X_live)
predict["Return Predictions SVM"] = sv_model.predict(X_live)


# In[24]:


drop_cols = ["BUY", "Returns","SHORT", "close_price", "return"]
predict.drop(feature_cols, axis = 1, inplace=True)
predict.drop(drop_cols, axis = 1, inplace = True)


# In[25]:


predict["LR Probability (-1)"] = logmodel.predict_proba(X_live)[:,0]
predict["LR Probability  (1)"] = logmodel.predict_proba(X_live)[:,1]
predict["RFC Probability (-1)"] = rfc.predict_proba(X_live)[:,0]
predict["RFC Probability  (1)"] = rfc.predict_proba(X_live)[:,1]


# In[26]:


predict


# 
# 
# * **NOTES:** :
#     * Causility Test between stock closing prices and lagged sentdex sentiment score. Check how many lags to apply.
#     * 1: https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/stattools.py
#     * 2: from statsmodels.tsa.stattools import grangercausalitytests

# In[70]:


def grangercausalitytests(x, maxlag, addconst=True, verbose=True):
    """four tests for granger non causality of 2 timeseries
    all four tests give similar results
    `params_ftest` and `ssr_ftest` are equivalent based on F test which is
    identical to lmtest:grangertest in R
    Parameters
    ----------
    x : array, 2d
        data for test whether the time series in the second column Granger
        causes the time series in the first column
    maxlag : integer
        the Granger causality test results are calculated for all lags up to
        maxlag
    verbose : bool
        print results if true
    Returns
    -------
    results : dictionary
        all test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        teststatistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.
    Notes
    -----
    TODO: convert to class and attach results properly
    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.
    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.
    'params_ftest', 'ssr_ftest' are based on F distribution
    'ssr_chi2test', 'lrtest' are based on chi-square distribution
    References
    ----------
    http://en.wikipedia.org/wiki/Granger_causality
    Greene: Econometric Analysis
    """
    from scipy import stats

    x = np.asarray(x)

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError("Insufficient observations. Maximum allowable "
                         "lag is {0}".format(int((x.shape[0] - int(addconst)) /
                                                 3) - 1))

    resli = {}

    for mlg in range(1, maxlag + 1):
        result = {}
        if verbose:
            print('\nGranger Causality')
            print('number of lags (no zero)', mlg)
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        #add constant
        if addconst:
            dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
        else:
            raise NotImplementedError('Not Implemented')
            #dtaown = dta[:, 1:mxlg]
            #dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        #print results
        #for ssr based tests see:
        #http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        #the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = ((res2down.ssr - res2djoint.ssr) /
                res2djoint.ssr / mxlg * res2djoint.df_resid)
        if verbose:
            print('ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                   ' df_num=%d' % (fgc1,
                                    stats.f.sf(fgc1, mxlg,
                                               res2djoint.df_resid),
                                    res2djoint.df_resid, mxlg))
        result['ssr_ftest'] = (fgc1,
                               stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                               res2djoint.df_resid, mxlg)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        if verbose:
            print('ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, '
                   'df=%d' % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg))
        result['ssr_chi2test'] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        #likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        if verbose:
            print('likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %
                   (lr, stats.chi2.sf(lr, mxlg), mxlg))
        result['lrtest'] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
                                   np.eye(mxlg, mxlg),
                                   np.zeros((mxlg, 1))))
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print('parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                   ' df_num=%d' % (ftres.fvalue, ftres.pvalue, ftres.df_denom,
                                    ftres.df_num))
        result['params_ftest'] = (np.squeeze(ftres.fvalue)[()],
                                  np.squeeze(ftres.pvalue)[()],
                                  ftres.df_denom, ftres.df_num)

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[130]:


beg_date = '2019-06-01'
end_date = '2019-07-01'


# In[131]:


stock = get_pricing("NDAQ", start_date = beg_date,
                    end_date = end_date,
                    frequency = 'daily')


# In[132]:


stock['close_price'].plot(label = "Closing Price", figsize = (20,8), c = 'blue', marker='o',
         markerfacecolor='red', markersize=6, linestyle = "--")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("AMAZON Clsong Prices 2018-07-01 till 2019-07-01)")
plt.legend(loc = 2);


# In[133]:


from quantopian.interactive.data.sentdex import sentiment as dataset

# import data operations
from odo import odo


# In[134]:


dataset.dshape


# In[135]:


ndaq = dataset[dataset.symbol == "NDAQ"]
ndaq_df = odo(ndaq.sort('asof_date'), pd.DataFrame)
plt.plot(ndaq_df.asof_date, ndaq_df.sentiment_signal, marker='.', linestyle='None', color='r')
# plt.plot(ndaq_df.asof_date, pd.rolling_mean(ndaq_df.sentiment_signal, 21))
# plt.plot(ndaq_df.asof_date, pd.rolling_mean(ndaq_df.sentiment_signal, 252))
plt.xlabel("As Of Date (asof_date)")
plt.ylabel("Sentiment")
plt.title("Sentdex Sentiment for NDAQ")
plt.legend(["Sentiment - Single Day"], loc=1)
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-4,7.5));


# In[136]:


initial = ndaq_df.index[ndaq_df.asof_date == beg_date][0]
end = ndaq_df.index[ndaq_df.asof_date == end_date][0] + 1


# In[137]:


my_test_df = pd.DataFrame()


# In[138]:


my_test_df["ClosePrice"] = stock['close_price']


# In[139]:


my_test_df["SentimentScore"] = ndaq_df['sentiment_signal'][initial:end]


# In[143]:


len(stock['close_price'])


# In[144]:


ndaq_df.head()


# In[147]:


stock.index


# In[ ]:




