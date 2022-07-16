import sys

from attr import attr

from Agri_Emmision_Time_Series_Code import *
from Agri_Emmision_Visualization_Code import *

import streamlit as st
import time
import numpy as np

import json
 
st.set_page_config(layout="wide")

strr='ClimateWatch - A Time Series Analysis Dashboard'
st.markdown("<h1 style='text-align: center; color: green;'>"+strr+"</h1>", unsafe_allow_html=True)

def agri_emmision_series(typee, paramss1, paramss2, paramss3, paramss4):
    if(typee=="emmision"):
        return return_series_emmision(paramss1, paramss2,
                                        paramss3, paramss4)
    elif(typee=="agriculture"):
        return return_series_agri(paramss1, paramss2,
                                paramss3)

def agri_emmision_multiple(typee, paramss1, paramss2, paramss3):
    if(typee=="emmision"):
        return return_multiplegas_emmision(paramss1, paramss2,
                                                paramss3)
    elif(typee=="agriculture"):
        return return_series_wrt_country_agri(paramss1, paramss2)

def agri_emmision_series(typee, paramss1, paramss2, paramss3, paramss4):
    if(typee=="emmision"):
        return return_series_emmision(paramss1, paramss2,
                                        paramss3, paramss4)
    elif(typee=="agriculture"):
        return return_series_agri(paramss1, paramss2,
                                paramss3)

def agrifun():
    # Opening JSON file
    col1,col2=st.columns(2)
    col1.header("Time Series Analysis: Agricultural Profile")
    metadatadf=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\CW_Agriculture_metadata.csv")
    col2.dataframe(metadatadf)
    jsonfilee=r'metadata.json'
    with open(jsonfilee) as json_file:
        data = json.load(json_file)
    col1,col2, col3,col4 = st.columns(4)
    typee = col1.selectbox(
        'Type of data',
        tuple(data.keys()))

    country = col2.selectbox(
        'Country',
        tuple(data[typee]['countries'])
    )

    filenamee = col3.selectbox(
        'File Name',
        tuple(data[typee]['files'].keys())
    )

    attribute = col4.selectbox(
        'Attribute',
        tuple(data[typee]['files'][filenamee])
    )

    vizbutton=col3.button("VISUALIZE")
    if(vizbutton):
        paramss=[typee, filenamee, country,attribute, '']
        finalseries=agri_emmision_series(typee, filenamee, country,
                                            attribute, '')
        data=return_series_wrt_country_agri(filenamee , country)
        seriess=agri_emmision_series(typee, filenamee, country, attribute, "")

        col1,col2 = st.columns(2)

        with col1.container():
            st.plotly_chart(generalplot_agri(finalseries, str(paramss)), use_container_width=True)
            with st.expander("See explanation"):
                st.write("""
                    This is a general plot visualizing the time series
                """)

            st.pyplot(detrend_agri(finalseries))
            with st.expander("See explanation"):
                st.write("""
                    Remove linear trend along axis from data.
                """)
            st.pyplot(rollmean_agri(finalseries, 5, str([filenamee, country, attribute])))
            with st.expander("See explanation"):
                st.write("Rolling mean of time series data with window size 5 to smooth the TS Data")

            st.pyplot(RNN_forecast(seriess))
            with st.expander("See explanation"):
                st.write("""
                    Time Series Forecasting done using Recurrent Neural Networks
                """)



        with col2.container():
            st.pyplot(seasonal_decom_agri(finalseries, str(paramss)))
            with st.expander("See explanation"):
                st.write("""
                    Seasonal Decomposition plot, decomposed with respect to various types
                """)

            st.pyplot(autocor_agri(finalseries))
            with st.expander("See explanation"):
                st.write("""
                    Autocorrelation Plot: used tool for checking randomness in a data set
                """)

            st.pyplot(autocor_compare_agri(finalseries))
            with st.expander("See explanation"):
                st.write("""
                    Comparison of Autocorrelation Plots provided by various libraries
                    : used tool for checking randomness in a data set
                """)

            st.pyplot(kalman_filter_multiple_data(data))
            
            with st.expander("See explanation"):
                st.write("""
                    Kalman filtering is an algorithm that produces estimates of unknown variables 
                    that tend to be more accurate than those based on a single measurement alone. 
                    In other words, Kalman filter takes time series as input and performs some kind 
                    of smoothing and denoising.
                """)

def emmisiontimeseriespart():
    st.header("Time Series Analysis: Historic Emmisions")
    jsonfilee=r'metadata.json'
    with open(jsonfilee) as json_file:
        data = json.load(json_file)

    col1,col2, col3,col4, col5 = st.columns(5)

    filename=col1.selectbox('File Name',
    tuple(data['emmision']['files'].keys()))

    country=col2.selectbox('Country',
    tuple(data['emmision']['countries']))

    sector=col3.selectbox('Sector',
    tuple(data['emmision']['files'][filename]['sectors']))

    gases=col4.selectbox('Gas',
    tuple(data['emmision']['files'][filename]['gases']))

    vizbutton=col5.button("VISUALIZE")

    if(vizbutton):
        filenamee, attribute=filename, sector
        paramss=[filenamee, country,attribute, gases]
        typee='emmision'
        finalseries=agri_emmision_series(typee, filenamee, country,
                                            attribute, gases)
        data=agri_emmision_multiple(typee, filenamee , country, attribute)
        seriess=agri_emmision_series(typee, filenamee, country, attribute, gases)

        col1,col2 = st.columns(2)

        with col1.container():
            st.plotly_chart(generalplot_agri(finalseries, str(paramss)), use_container_width=True)
            with st.expander("See explanation"):
                st.write("""
                    This is a general plot visualizing the time series
                """)

            st.pyplot(detrend_agri(finalseries))
            with st.expander("See explanation"):
                st.write("""
                    Remove linear trend along axis from data.
                """)

            st.pyplot(rollmean_agri(finalseries, 5, str([filenamee, country, attribute])))
            with st.expander("See explanation"):
                st.write('Rolling mean of time series data with window size 5 to smooth the TS Data')

            st.pyplot(RNN_forecast(seriess))
            with st.expander("See explanation"):
                st.write("""
                    Time Series Forecasting done using Recurrent Neural Networks
                """)



        with col2.container():
            st.pyplot(seasonal_decom_agri(finalseries, str(paramss)))
            with st.expander("See explanation"):
                st.write("""
                    Seasonal Decomposition plot, decomposed with respect to various types
                """)

            st.pyplot(autocor_agri(finalseries))
            with st.expander("See explanation"):
                st.write("""
                    Autocorrelation Plot: used tool for checking randomness in a data set
                """)

            st.pyplot(autocor_compare_agri(finalseries))
            with st.expander("See explanation"):
                st.write("""
                    Comparison of Autocorrelation Plots provided by various libraries
                    : used tool for checking randomness in a data set
                """)

            st.pyplot(kalman_filter_multiple_data(data))
            
            with st.expander("See explanation"):
                st.write("""
                    Kalman filtering is an algorithm that produces estimates of unknown variables 
                    that tend to be more accurate than those based on a single measurement alone. 
                    In other words, Kalman filter takes time series as input and performs some kind 
                    of smoothing and denoising.
                """)

def emmisionviz():
    st.header("Emmisions Data Sectors Comparison WRT Gas")

    jsonfilee=r'metadata.json'
    with open(jsonfilee) as json_file:
        data = json.load(json_file)

    col1,col2, col3=st.columns(3)

    filename=col1.selectbox('File Name',
    tuple(data['emmision']['files'].keys()))

    gases=col2.selectbox('Gas',
    tuple(data['emmision']['files'][filename]['gases']))

    year=col3.selectbox('Year',
    tuple([str(i) for i in range(2000, 2017)]))

    areas = col1.multiselect("Select Areas (Max 4)", 
        list(data['emmision']['countries']),["USA", "CAN", "AUS", "MEX"])

    typee = col2.multiselect("Select Sector (Max 4)", 
        list(data['emmision']['files'][filename]['sectors']))

    vizbutton=col3.button("VISUALIZE")
    # col1,col2=st.columns(2)
    if(vizbutton):
        if((len(areas)<=4) & (len(typee)<=4)):
            df=return_vizdata_emmision(filename, "",  "", gases)
            finalseries=radar_bar_dict_agri(df, areas, typee, year)
            # st.info(finalseries)
            st.plotly_chart(return_radar_bar_agri(finalseries, True), use_container_width=True)
            st.plotly_chart(return_radar_bar_agri(finalseries, False), use_container_width=True)
        else:
            st.warning("Please select only 4 values in Sector and Areas")

    col1,col2, col3=st.columns(3)
    col1.header("Violin Plot")
    typee_viol=col2.selectbox("Select a Sector", 
        list(data['emmision']['files'][filename]['sectors']))
    vizbutton2=col3.button("VISUALIZE Violin Plot")
    if(vizbutton2):
        df=return_vizdata_emmision(filename, "",  "", gases)
        col2.pyplot(violin_agri(df, areas, typee_viol))


    col1,col2, col3=st.columns(3)
    col1.header("Treemap Plot")
    options= col2.multiselect("Select Sector (Max 2)", 
        list(data['emmision']['files'][filename]['sectors']),['Building',
        'Waste '])
    vizbutton2=col3.button("VISUALIZE Treemap Plot")
    if(vizbutton2):
        df=return_vizdata_emmision(filename, "",  "", gases)
        asd=treemap_agri_data(df, options, year)
        st.plotly_chart(get_treemap_agri(asd, options, year), use_container_width=True)

    col1,col2, col3=st.columns(3)
    col1.header("World Map Plot")
    options= col2.selectbox("Select Sector for world map", 
        tuple(data['emmision']['files'][filename]['sectors']))
    vizbutton2=col3.button("VISUALIZE World Map")
    if(vizbutton2):
        df=return_vizdata_emmision(filename, "",  "", gases)
        st.plotly_chart(worldmap_timeseries(df, options), use_container_width=True)

def adaptationviz():
    col1,col2=st.columns(2)
    col1.header("Adaptation Vizualisations: Parallel Categories")
    col2.dataframe(pd.read_csv("data\ClimateWatch_Adaptation\CW_adaptation_metadata.csv"))
    filename=r"data\ClimateWatch_Adaptation\CW_adaptation.csv"
    col1, col2, col3, col4, col5, col6, col7, col8=st.columns(8)
    poverty_14=col1.number_input(label= "Division for Poverty Rate", value = 4, min_value =1)
    climate_risks=col2.number_input(label= "Division for climate_risks", value = 4, min_value =1)
    climate_risks_rank=col3.number_input(label= "Division for climate_risks_rank", value = 4, min_value =1)
    vulnerability=col4.number_input(label= "Division for vulnerability", value = 4, min_value =1)
    vulnerability_rank=col5.number_input(label= "Division for vulnerability_rank", value = 4, min_value =1)
    readiness=col6.number_input(label= "Division for readiness", value = 4, min_value =1)
    readiness_rank=col7.number_input(label= "Division for readiness_rank", value = 4, min_value =1)

    partition_dict={"poverty_14":poverty_14,"climate_risks":climate_risks,
    "climate_risks_rank":climate_risks_rank,"vulnerability":vulnerability,
    "vulnerability_rank":vulnerability_rank,"readiness":readiness,"readiness_rank":readiness_rank}

    viznow=col8.button("VISUALIZE Parallel Category Plot")
    if(viznow):
        st.plotly_chart(parallel_cate_adap(filename, partition_dict), use_container_width=True)

    st.header("3D PLOT FOR Factors comparison")
    col1, col2, col3, col4, col5=st.columns(5)
    attr1=col1.selectbox("Attribute 1: X", tuple(list(partition_dict.keys())))
    attr2=col2.selectbox("Attribute 2: Y", tuple(list(partition_dict.keys())))
    attr3=col3.selectbox("Attribute 3: Z", tuple(list(partition_dict.keys())))
    attr4=col4.selectbox("Attribute 4: COLOR", tuple(list(partition_dict.keys())))
    viznow=col5.button("VISUALIZE 3D PLOT FOR Factors comparison")
    attributes=[attr1,attr2, attr3, attr4 ]
    if(viznow):
        df=pd.read_csv(filename).drop(["wb_urls"], axis=1).fillna(0)

        fig = px.scatter_3d(df, x=attributes[0], y=attributes[1], z=attributes[2],
                            color=attributes[3], hover_name ="country")
        st.plotly_chart(fig, use_container_width=True, height=3000)


# adaptationviz()

def agriviz():
    # st.header("Agricultural Profiles Data Comparison")
    col1,col2=st.columns(2)
    col1.header("Agricultural Profiles Data Comparison")
    metadatadf=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\CW_Agriculture_metadata.csv")
    col2.dataframe(metadatadf)
    jsonfilee=r'metadata.json'
    with open(jsonfilee) as json_file:
        data = json.load(json_file)

    col1,col2, col3=st.columns(3)

    filename=col1.selectbox('File Name',
    tuple(data['agriculture']['files'].keys()),tuple(data['agriculture']['files'].keys()).index("CW_Agriculture_emissions.csv"))

    year=col2.selectbox('Year',
    tuple([str(i) for i in range(2000, 2017)]))

    areas = col1.multiselect("Select Areas (Max 4)", 
        list(data['agriculture']['countries']),["USA", "CAN", "AUS", "MEX"])

    typee = col2.multiselect("Select Sector (Max 4)", 
        list(data['agriculture']['files'][filename]))

    df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\"+filename)

    vizbutton=col3.button("VISUALIZE")
    # col1,col2=st.columns(2)
    if(vizbutton):
        if((len(areas)<=4) & (len(typee)<=4)):
            finalseries=radar_bar_dict_agri(df, areas, typee, year)
            # st.info(finalseries)
            st.plotly_chart(return_radar_bar_agri(finalseries, True), use_container_width=True)
            st.plotly_chart(return_radar_bar_agri(finalseries, False), use_container_width=True)
        else:
            st.warning("Please select only 4 values in Sector and Areas")

    col1,col2, col3=st.columns(3)
    col1.header("Violin Plot")
    typee_viol=col2.selectbox("Select Sector", 
        tuple(data['agriculture']['files'][filename]))

    vizbutton2=col3.button("VISUALIZE Violin Plot")
    if(vizbutton2):
        df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\"+filename)
        col2.pyplot(violin_agri(df, areas, typee_viol))


    col1,col2, col3=st.columns(3)
    col1.header("Treemap Plot")
    options= col2.multiselect("Select Sector (Max 2)", 
        list(data['agriculture']['files'][filename]))
    vizbutton2=col3.button("VISUALIZE Treemap Plot")
    if(vizbutton2):
        df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\"+filename)
        asd=treemap_agri_data(df, options, year)
        st.plotly_chart(get_treemap_agri(asd, options, year), use_container_width=True)

    col1,col2, col3=st.columns(3)
    col1.header("World Map Plot")
    options= col2.selectbox("Select Sector for world map plot", 
        tuple(data['agriculture']['files'][filename]))
    vizbutton2=col3.button("VISUALIZE World Map")
    if(vizbutton2):
        df=pd.read_csv(r"data\ClimateWatch_AgricultureProfile\\"+filename)
        st.plotly_chart(worldmap_timeseries(df, options), use_container_width=True)


page_names_to_funcs = {
    "TS Analysis:Agricultural Profile": agrifun,
    "TS Analysis: Historic Emmisions": emmisiontimeseriespart,
    "ViZ Analysis:Agricultural Profile":agriviz,
    "ViZ Analysis:Historic Emmisions": emmisionviz,
    "ViZ Analysis:Adaptation Data": adaptationviz

}
selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
try:
    page_names_to_funcs[selected_page]()
except:
    st.error("DATA NOT AVAILABLE")