#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(0, 'E:\BOOKSDUMP2\ClimateWatch_FinalNoteBooks\Time Series Code\\')


# In[ ]:


from Agri_Emmision_Time_Series_Code import *


# In[ ]:


from Agri_Emmision_Visualization_Code import *




df=return_vizdata_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
areas=["USA", "CAN", "AUS", "MEX"]
typee=['Building',
       'Waste ',
        'Transportation ',
        'Other Fuel Combustion ']
year="2015"
finalseries=radar_bar_dict_agri(df, areas, typee, year)
finalseries


# # Radar/Bar Charts

# In[ ]:


return_radar_bar_agri(finalseries, True)
return_radar_bar_agri(finalseries, False)


# # Violin Plot

# In[ ]:


df=return_vizdata_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
violin_agri(df, areas, 'Other Fuel Combustion ')


# # TreeMap

# In[ ]:


# options=["total_emissions_agr_9", "total_emissions_agr_7"]
year="2015"
options=['Building',
       'Waste ']
df=return_vizdata_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')
asd=treemap_agri_data(df, options, year)
get_treemap_agri(asd, options, year)


# # Parallel Categories: STATIC DATA

# In[ ]:
############################################################## NEEDFINISH

filename="data\ClimateWatch_Adaptation\CW_adaptation.csv"
partition_dict={"poverty_14":4,"climate_risks":4,"climate_risks_rank":4,"vulnerability":4,"vulnerability_rank":4,"readiness":4,"readiness_rank":4}
parallel_cate_adap(filename, partition_dict)


# # 3D Scatter: STATIC DATA

# In[ ]:


attributes=['poverty_14','climate_risks_rank', 'vulnerability_rank', 'readiness_rank' ]
df=pd.read_csv(r"data\ClimateWatch_Adaptation\CW_adaptation.csv").drop(["wb_urls"], axis=1).fillna(0)


# In[ ]:


fig = px.scatter_3d(df, x=attributes[0], y=attributes[1], z=attributes[2],
                    color=attributes[3], hover_name ="country")
fig.show()


# # World Map

# In[ ]:


option='Other Fuel Combustion '
df=return_vizdata_emmision("CW_HistoricalEmissions_CAIT.csv", "AFG",  "Total excluding LUCF", 'All GHG')

worldmap_timeseries(df, option)


# In[ ]:





# In[ ]:





# # Radar/Bar Charts: AGRI

# In[18]:


df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\CW_Agriculture_emissions.csv")
areas = ['USA', 'CAN', "AUS", "MEX"]#, "IND"] 

typee=['total_emissions_agr_1',
'total_emissions_agr_10',
'total_emissions_agr_11',
'total_emissions_agr_12',
'total_emissions_agr_13']

year="2011"
finalseries=radar_bar_dict_agri(df, areas, typee, year)
finalseries


# In[ ]:


return_radar_bar_agri(finalseries, True)
return_radar_bar_agri(finalseries, False)


# # Violin Plot: AGRI

# In[ ]:


df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\CW_Agriculture_emissions.csv")

violin_agri(df, areas, "total_emissions_agr_12")


# # TreeMap: AGRI

# In[ ]:


options=["total_emissions_agr_9", "total_emissions_agr_7"]
year="1962"
df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\CW_Agriculture_emissions.csv")

get_treemap_agri(treemap_agri_data(df, options, year), options, year)


# In[ ]:


filename="E:\ClimateWatch Data\ClimateWatch_Adaptation\CW_adaptation.csv"
partition_dict={"poverty_14":4,"climate_risks":4,"climate_risks_rank":4,"vulnerability":4,"vulnerability_rank":4,"readiness":4,"readiness_rank":4}
parallel_cate_adap(filename, partition_dict)


# In[ ]:


attributes=['poverty_14','climate_risks_rank', 'vulnerability_rank', 'readiness_rank' ]
df=pd.read_csv(r"E:\ClimateWatch Data\ClimateWatch_Adaptation\CW_adaptation.csv").drop(["wb_urls"], axis=1).fillna(0)


# In[ ]:


fig = px.scatter_3d(df, x=attributes[0], y=attributes[1], z=attributes[2],
                    color=attributes[3], hover_name ="country")
fig.show()


# # World Map: AGRI

# In[24]:


option="total_emissions_agr_9"
filename="CW_Agriculture_emissions.csv"
df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\CW_Agriculture_emissions.csv")

worldmap_timeseries(df, option)


# In[ ]:




