#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pprint import pprint

from Agri_Emmision_Time_Series_Code import *




# # General Plot

# In[3]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
generalplot_agri(finalseries, str(params))


# # Seasonal Decomposition Plot

# In[4]:


# finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
# seasonal_decom_agri(finalseries, str([agri_files[1], "AFG", "total_emissions_agr_9"]))

params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
seasonal_decom_agri(finalseries, str(params))


# # Detrending using Scipy Signal

# In[5]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
detrend_agri(finalseries)


# # Autocorrelation Plot

# In[6]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
autocor_agri(finalseries)


# # Cyclic Variation

# In[7]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
cyclicvar_agri(finalseries)


# # Double Exponential Smoothing
# 

# In[8]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
double_expsmooth_agri(finalseries)


# # Partial AutoCorrelation

# In[9]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
partialautocor_agri(finalseries)


# # AutoRegressive Model: ARIMA

# In[10]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
arima_agri(finalseries)


# # Anomaly Detection

# In[11]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
anomaly_agri(finalseries)


# #  Rolling mean with window size convolve

# In[12]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
rollmean_agri(finalseries, 5, str(params))


# # Comparing AutoCorrelations

# In[13]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
autocor_compare_agri(finalseries)


# # Prediction using RNN

# In[14]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
finalseries=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
RNN_forecast(finalseries)


# # Univariate Multi-step Vector-output 1d Cnn Example

# In[15]:


# demonstrate prediction
params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
seriess=return_series_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
no_pred=23
ourmodel=univariate_1d_cnn_model(seriess, no_pred)
x_input = array([70+i*10 for i in range(no_pred)])
x_input = x_input.reshape((1, no_pred, 1))
yhat = ourmodel.predict(x_input, verbose=0)
print(yhat)


# # Return All Feature Data With Respect to Countries

# # KALMAN FILTERS

# In[16]:


params=["CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG']
data=return_multiplegas_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF")
kalman_filter_multiple_data(data)


# # FOR AGRI

# # General Plot

# In[19]:


agri_files=["ASD", "CW_Agriculture_emissions.csv"]


# In[20]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
generalplot_agri(finalseries, str([agri_files[1], "AFG", "total_emissions_agr_9"]))


# # Seasonal Decomposition Plot

# In[21]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
seasonal_decom_agri(finalseries, str([agri_files[1], "AFG", "total_emissions_agr_9"]))


# # Detrending using Scipy Signal

# In[22]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
detrend_agri(finalseries)


# # Autocorrelation Plot

# In[23]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
autocor_agri(finalseries)


# # Cyclic Variation

# In[24]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
cyclicvar_agri(finalseries)


# # Double Exponential Smoothing
# 

# In[25]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
double_expsmooth_agri(finalseries)


# # Partial AutoCorrelation

# In[26]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
partialautocor_agri(finalseries)


# # AutoRegressive Model: ARIMA

# In[27]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")

arima_agri(finalseries)


# # Anomaly Detection

# In[28]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
anomaly_agri(finalseries)


# #  Rolling mean with window size convolve

# In[29]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
rollmean_agri(finalseries, 5, str([agri_files[1], "AFG", "total_emissions_agr_9"]))


# # Comparing AutoCorrelations

# In[30]:


finalseries=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
autocor_compare_agri(finalseries)


# # KALMAN FILTERS

# In[31]:


data=return_series_wrt_country_agri(agri_files[1] , "USA")
kalman_filter_multiple_data(data)


# # Prediction using RNN

# In[32]:


seriess=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
RNN_forecast(seriess)


# # Univariate Multi-step Vector-output 1d Cnn Example

# In[33]:


# demonstrate prediction
seriess=return_series_agri(agri_files[1], "AFG", "total_emissions_agr_9")
no_pred=23
ourmodel=univariate_1d_cnn_model(seriess, no_pred)
x_input = array([70+i*10 for i in range(no_pred)])
x_input = x_input.reshape((1, no_pred, 1))
yhat = ourmodel.predict(x_input, verbose=0)
print(yhat)

