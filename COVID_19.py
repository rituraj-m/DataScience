#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
import pandas as pd 
import os

# Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import pycountry
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

from pywaffle import Waffle
from pymongo import MongoClient 

py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
plt.style.use("fivethirtyeight")

plt.rcParams['figure.figsize'] = 8, 5


# # Analysis of present condition in India COVID19

# In[14]:


# creation of MongoClient 
client=MongoClient() 
  
# Connect with the portnumber and host 
client = MongoClient("mongodb://localhost:27017/") 
  
# Access database 
COVIDdatabase = client['COVID19'] 
  
# Access collection of the database 
Covid_India=COVIDdatabase['Covid_India']
Location = COVIDdatabase['Indian_Coordinates']
Cases = COVIDdatabase['per_day_cases']


# In[31]:


try:
    Covid_Indiadata = Covid_India.find()
    Locationdata = Location.find()
    PerdayCases = Cases.find()
    Coviddata = pd.DataFrame(list(Covid_Indiadata))
    Loc_data = pd.DataFrame(list(Locationdata))
    casesdata = pd.DataFrame(list(PerdayCases))
    Coviddata.drop(['_id','S_No'], axis = 1, inplace = True)
    #print(Coviddata)
    #print(Loc_data)
    #print(casesdata)
    print ("COVID19 dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")


# In[16]:


Coviddata['Total cases'] = Coviddata['Total Confirmed cases (Indian National)'] + Coviddata['Total Confirmed cases ( Foreign National )']
Coviddata['Active cases'] = Coviddata['Total cases'] - (Coviddata['Cured/Discharged/Migrated'] + Coviddata['Deaths'])
print(f'Total number of Confirmed COVID 2019 cases across India:', Coviddata['Total cases'].sum())
print(f'Total number of Active COVID 2019 cases across India:', Coviddata['Active cases'].sum())
print(f'Total number of Cured/Discharged/Migrated COVID 2019 cases across India:', Coviddata['Cured/Discharged/Migrated'].sum())
print(f'Total number of Deaths due to COVID 2019  across India:', Coviddata['Deaths'].sum())
print(f'Total number of States/UTs affected:', len(Coviddata['Name of State / UT']))


# # Distribution of Cases in India

# In[17]:


def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: red' if v else '' for v in is_max]



#df.style.apply(highlight_max,subset=['Total Confirmed cases (Indian National)', 'Total Confirmed cases ( Foreign National )'])
Coviddata.style.apply(highlight_max,subset=['Cured/Discharged/Migrated', 'Deaths','Total cases','Active cases'])


# # State/Union Territories wise number of Covid-19 cases

# In[18]:


statedata = Coviddata.groupby('Name of State / UT')['Active cases'].sum().sort_values(ascending=False).to_frame()
statedata.style.background_gradient(cmap='Reds')


# In[19]:


fig = px.bar(Coviddata.sort_values('Active cases', ascending=False).sort_values('Active cases', ascending=True), 
             x="Active cases", y="Name of State / UT", title='Total Active Cases', text='Active cases', orientation='h', 
             width=1000, height=700, range_x = [0, max(Coviddata['Active cases'])])
fig.update_traces(marker_color='#cf6a56', opacity=0.8, textposition='inside')

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.show()


# In[20]:


df_condensed = pd.DataFrame([Coviddata['Active cases'].sum(),Coviddata['Cured/Discharged/Migrated'].sum(),Coviddata['Deaths'].sum()],columns=['Cases'])
df_condensed.index=['Active cases','Recovered','Death']
df_condensed


fig = plt.figure(
    FigureClass=Waffle, 
    rows=5,
    values=df_condensed['Cases'],
    labels=list(df_condensed.index),
    figsize=(10, 3),
    legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)}
)


# In[21]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("National Cases","Foreign Cases"))

temp = Coviddata.sort_values('Total Confirmed cases (Indian National)', ascending=False).sort_values('Total Confirmed cases (Indian National)', ascending=False)

fig.add_trace(go.Bar( y=temp['Total Confirmed cases (Indian National)'], x=temp["Name of State / UT"],  
                     marker=dict(color=temp['Total Confirmed cases (Indian National)'], coloraxis="coloraxis")),
              1, 1)
                     
temp1 = Coviddata.sort_values('Total Confirmed cases ( Foreign National )', ascending=False).sort_values('Total Confirmed cases ( Foreign National )', ascending=False)

fig.add_trace(go.Bar( y=temp1['Total Confirmed cases ( Foreign National )'], x=temp1["Name of State / UT"],  
                     marker=dict(color=temp1['Total Confirmed cases ( Foreign National )'], coloraxis="coloraxis")),
              1, 2)                     
                     

fig.update_layout(coloraxis=dict(colorscale='rdbu'), showlegend=False,title_text="National vs Foreign Cases",plot_bgcolor='rgb(250, 242, 242)')
fig.show()


# # Nationality

# In[22]:


colors = ['#1f77b4', '#1f7b48']

indian = Coviddata['Total Confirmed cases (Indian National)'].sum()
foreign = Coviddata['Total Confirmed cases ( Foreign National )'].sum()
fig = go.Figure(data=[go.Pie(labels=['Indian','Foreign Nationals'],
                             values= [indian,foreign],hole =.3)])
                          

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))
fig.show()


# In[27]:


Coviddata_full = pd.merge(Loc_data,Coviddata,on='Name of State / UT')
map = folium.Map(location=[15, 60], zoom_start=3.5,tiles='Stamen Toner')

for lat, lon, value, name in zip(Coviddata_full['Latitude'], Coviddata_full['Longitude'], Coviddata_full['Active cases'], Coviddata_full['Name of State / UT']):
    folium.CircleMarker([lat, lon],
                        radius=value*0.7,
                        popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Active Cases</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.3 ).add_to(map)
map


# In[25]:


f, ax = plt.subplots(figsize=(12, 8))
data = Coviddata_full[['Name of State / UT','Total cases','Cured/Discharged/Migrated','Deaths']]
data.sort_values('Total cases',ascending=False,inplace=True)
sns.set_color_codes("pastel")
sns.barplot(x="Total cases", y="Name of State / UT", data=data,
            label="Total", color="r")

sns.set_color_codes("muted")
sns.barplot(x="Cured/Discharged/Migrated", y="Name of State / UT", data=data,
            label="Recovered", color="g")


# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 35), ylabel="",
       xlabel="Cases")
sns.despine(left=True, bottom=True)


# 

# In[ ]:





# In[ ]:





# In[ ]:




