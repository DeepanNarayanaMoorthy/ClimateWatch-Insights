import pandas as pd
# import time
# import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib as mpl
import numpy as np 
# import re
import matplotlib, random
import plotly.express as px
import plotly.graph_objects as go

import bisect
# Import settings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import numpy as np
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint
import traceback

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# import plotly.express as px

# from sklearn.preprocessing import MinMaxScaler





def return_vizdata_emmision(emm_file, country, factor,  gas):
    try:
        df=pd.read_csv(str(r"data\ClimateWatch_HistoricalEmissions\\") + emm_file)

        for j in range(len(df.columns)):
            # print(df.columns[j].strip(" ").lower())
            if (df.columns[j].strip(" ").lower()=="country"):
                countrycol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ").lower()=="sector"):
                sectorcol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ").lower()=="gas"):
                gascol=j
                break

        colls=list(df.columns)
        colls[countrycol]="area"
        colls[sectorcol]="short names"
        colls[gascol]="gas"
        df.columns=[i.lower() for i in colls]
        # pprint(df)
        df=df.loc[df["gas"] == gas].drop(["gas"], axis=1)
        try:
            df=df.drop(["source"], axis=1)
        except:
            pass
        return(df)
    except Exception as e:
        return "Data Incorrect", str(e)





def radar_bar_dict_agri(df, areas, typee, year):
    # df=pd.read_csv(agri_file)
    
    for j in range(len(df.columns)):
        if (df.columns[j].strip(" ")=="short names"):
            shortnamecol=j
            break
            
    for j in range(len(df.columns)):
        if (df.columns[j].strip(" ")=="area"):
            areacol=j
            break
    df=df.loc[(df[df.columns[areacol]].isin(areas)) & 
              (df[df.columns[shortnamecol]].isin(typee))].fillna(0)[[df.columns[areacol], df.columns[shortnamecol], year]]
    dictt={}
    # dictt["Countries"]=typee
    for i in typee:
        dictt[i]={}
    for i in typee:
        for j in areas:
            if not(df.loc[(df[df.columns[areacol]]==j) & (df[df.columns[shortnamecol]]==i)][year].empty):
                dictt[i][j]=list(df.loc[(df[df.columns[areacol]]==j) & (df[df.columns[shortnamecol]]==i)][year])[0]
            else:
                dictt[i][j]=0
    return(dictt)

        


def return_radar_bar_agri(dictt, isradar):
    typee=list(dictt.keys())
    areas=list(dictt[typee[0]].keys())
    
    ncols=3
    if len(areas)%3==0:
        nrows=len(areas)/3
    else:
        nrows=int(len(areas)/3)+1
    if(isradar):
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=[[{"type": "polar"} for _ in range(ncols)] for _ in range(nrows)],
        )
        positions=[]
        for i in range(nrows):
            for j in range(ncols):
                positions.append([i+1,j+1])

        for i in range(len(typee)):
            fig.add_trace(go.Scatterpolar(
                  r=[dictt[typee[i]][j] for j in areas],
                  theta=areas,
                  fill='toself',
                  name=typee[i]
            ),
                         row= positions[i][0],
                         col= positions[i][1])
        fig.update_layout(height=600, width=1300, title_text="Side By Side Subplots")
        return(fig)
        # fig.show()
    else:
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=[[{"type": "bar"} for _ in range(ncols)] for _ in range(nrows)],
        )
        positions=[]
        for i in range(nrows):
            for j in range(ncols):
                positions.append([i+1,j+1])

        for i in range(len(typee)):
            fig.add_trace(go.Bar(
                  y=[dictt[typee[i]][j] for j in areas],
                x=typee,
                  name=typee[i]
            ),
                         row= positions[i][0],
                         col= positions[i][1])
        fig.update_layout(height=600, width=1300, title_text="Side By Side Subplots")
        return(fig)
        # fig.show()
    



def violin_agri(df, areas, feature):
    try:
        # df=pd.read_csv(agri_file)

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ")=="short names"):
                shortnamecol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ")=="area"):
                areacol=j
                break
        mydata=df.loc[(df[df.columns[areacol]].isin(areas)) & (df[df.columns[shortnamecol]]==feature)].dropna(axis="columns").T
        mydata.columns = mydata.iloc[0]
        mydata = mydata[2:]
        mydata=mydata.reset_index().drop(["index"], axis=1)

        groups=[]

        for i in range(len(mydata.columns)):
            groups.append(mydata[mydata.columns[i]].tolist())

        group_name_list=[]
        group_scores_list=[]
        for i in range(len(groups)):
            group_name_list=group_name_list+[mydata.columns[i]]* len(groups[i])
            group_scores_list=group_scores_list+groups[i]

        data = pd.DataFrame({'Areas': group_name_list, 
                             'Values':group_scores_list})
        plt.figure(dpi=150)
        sns.set_style('whitegrid')
        sns.violinplot('Areas', 'Values', data=data)
        sns.despine(left=True, right=True, top=True)
        plt.title('Violin Plot for: '+str(areas)+" for the feature: "+feature)
        return(plt)
        # plt.show()
    except Exception as e:
        return "Data Insufficient", str(traceback.format_exc())


def treemap_agri_data(df, options, year):
    # try:
        # df=pd.read_csv(agrifile)

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ")=="short names"):
                shortnamecol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ")=="area"):
                areacol=j
                break

        df=df.loc[df[df.columns[shortnamecol]].isin(options)]
        df=df[[df.columns[areacol], df.columns[shortnamecol], year]]
        colss=list(df.columns)
        colss[areacol]="area"
        colss[shortnamecol]="short names"
        df.columns=colss
        return df
    # except Exception as e:
    #     return "Data Insufficient", str(e)




def get_treemap_agri(df, options, year):
    try:
        col1=list(set(list(df["area"])))

        col2=[]
        for i in col1:
            if not(df[(df["area"]==i) & (df["short names"]==options[0])]).empty:
                valuee=list(df[(df["area"]==i) & (df["short names"]==options[0])][year])[0]
                if(valuee==0):
                    col2.append(1)
                else:
                    col2.append(1)
            else:
                col2.append(1)

        col3=[]
        for i in col1:
            if not(df[(df["area"]==i) & (df["short names"]==options[1])]).empty:
                col3.append(list(df[(df["area"]==i) & (df["short names"]==options[1])][year])[0])
            else:
                col3.append(1)

        heat=pd.DataFrame({"Country":col1,
                     options[0]: col2,
                     options[1]: col3})

        main_df=df.copy()
        plot_df = px.data.gapminder().reset_index().drop(["index"], axis=1)
        pl_con=list(set(list(plot_df["iso_alpha"])))
        my_con=list(set(list(main_df["area"])))

        comm=list(set(my_con).intersection(pl_con))


        heatfinal=heat[heat["Country"].isin(comm)].reset_index().drop(["index"], axis=1)

        col1list=[]
        col2list=[]
        contlist=[]
        countryname=[]
        comm_list=[]
        for i in comm:
            try:
                col1list_elt=heatfinal.iloc[heatfinal[heatfinal['Country'] == i].index[0], heatfinal.columns.get_loc(options[0])]
                col2list_elt=heatfinal.iloc[heatfinal[heatfinal['Country'] == i].index[0], heatfinal.columns.get_loc(options[1])]
                contlist_elt=plot_df.iloc[plot_df[plot_df['iso_alpha'] == i].index[0], plot_df.columns.get_loc('continent')]
                countryname_elt=plot_df.iloc[plot_df[plot_df['iso_alpha'] == i].index[0], plot_df.columns.get_loc('country')]
                col1list.append(col1list_elt)
                col2list.append(col2list_elt)
                contlist.append(contlist_elt)
                countryname.append(countryname_elt)
                comm_list.append(i)
            except:
                pass


        heatfinal=pd.DataFrame({"country":countryname,
                     "continent":contlist,
                     "iso_alpha":comm_list,
                     options[0]:col1list,
                     options[1]:col2list})


        fig = px.treemap(heatfinal, path=[px.Constant("world"), 'continent', 'country'], values=options[0],
                          color=options[1], hover_data=['iso_alpha'],
                          color_continuous_scale='RdBu',
                          color_continuous_midpoint=np.average(heatfinal[options[1]], weights=heatfinal[options[0]]))
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        return(fig)
        # fig.show()
    except Exception as e:
        return "Data Insufficient", str(e)


def parallel_cate_adap(filenamee, partition_dict):
    try:
        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


        hex_colors_dic = {}
        rgb_colors_dic = {}
        hex_colors_only = []
        for name, hex in matplotlib.colors.cnames.items():
            hex_colors_only.append(hex)
            hex_colors_dic[name] = hex
            rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

        def intreval_list_finder(listt, intr_cnt):
            listt_2=list(split(sorted(set(listt)), intr_cnt))
            lst=[0]
            for i in listt_2:
                i.sort()
                lst.append(i[len(i)-1])
            intreval_list=[str(round(lst[i], 2))+"-"+str(round(lst[i+1], 2)) for i in range(len(lst)-1)]
            return [intreval_list[bisect.bisect_left(lst, i)-1] for i in listt]

        df=pd.read_csv(filenamee).drop(["wb_urls"], axis=1).fillna(0)

        intr_dict=partition_dict
        for i in ["poverty_14","climate_risks","climate_risks_rank","vulnerability","vulnerability_rank","readiness","readiness_rank"]:
            df[i]=intreval_list_finder(list(df[i]), intr_dict[i])

        config = dict({'scrollZoom': True})

        fig = px.parallel_categories(df, width=3500, height=900)#, color=[random.choice(hex_colors_only) for i in range(len(df))])
        return(fig)
        # fig.show(config=config)
    except Exception as e:
        return "Data Insufficient", str(e)



def worldmap_timeseries(df, option):
    try:
        # df=pd.read_csv(filename)

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ")=="short names"):
                shortnamecol=j
                break

        for j in range(len(df.columns)):
            if (df.columns[j].strip(" ")=="area"):
                areacol=j
                break
        colls=list(df.columns)
        colls[shortnamecol]="short names"
        colls[areacol]='area'
        df.columns=colls

        plot_df = px.data.gapminder().reset_index().drop(["index"], axis=1)
        pl_con=list(set(list(plot_df["iso_alpha"])))
        my_con=list(set(list(df["area"])))
        comm=list(set(my_con).intersection(pl_con))


        df=df.loc[df["short names"]==option].drop("short names", axis="columns")
        df=df.set_index("area")
        cols=list(df.columns)
        country=[]
        year=[]
        value=[]

        for i in comm:
            for j in cols:
                try:
                    vall=df.loc[[i],[j]].iat[0,0]
                    country.append(i)
                    year.append(int(j))
                    value.append(vall)
                except:
                    country.append(i)
                    year.append(int(j))
                    value.append(0)

        worlddf=pd.DataFrame({"country":country ,
                     "year":year ,
                     "value":value })

        fig = px.choropleth(worlddf,
                            locations ="country",
                            color ="value",
                            hover_name ="country", 
                            color_continuous_scale = px.colors.sequential.Plasma,
                            # color_continuous_scale = px.colors.cyclical.Earth_r,
                            scope ="world",
                            animation_frame ="year",
                           projection="natural earth",
                            width=1600, height=1000)
        return(fig)
        # fig.show()
    except Exception as e:
        return "Data Insufficient", str(e)




