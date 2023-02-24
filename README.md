---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.12
  nbformat: 4
  nbformat_minor: 5
  papermill:
    default_parameters: {}
    duration: 21.671949
    end_time: "2023-02-24T02:31:46.173939"
    environment_variables: {}
    input_path: \_\_notebook\_\_.ipynb
    output_path: \_\_notebook\_\_.ipynb
    parameters: {}
    start_time: "2023-02-24T02:31:24.501990"
    version: 2.3.4
---

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.0205e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:31.892359&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:31.882154&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

![](https://npr.brightspotcdn.com/legacy/sites/kosu/files/201706/airbnb.png)

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:9.033e-3,&quot;end_time&quot;:&quot;2023-02-24T02:31:31.910633&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:31.901600&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<center><font size=18px;> Airbnb in New York </font></center>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:9.308e-3,&quot;end_time&quot;:&quot;2023-02-24T02:31:31.931973&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:31.922665&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<center>

<font size=3px;>Airbnb is a popular platform for people who want to
travel and stay in a comfortable and affordable place. In this data
analysis project, we will be exploring the Airbnb listings in New York
City.</font><center>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:8.25e-3,&quot;end_time&quot;:&quot;2023-02-24T02:31:31.950499&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:31.942249&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
    📌 &nbsp; The dataset includes information about the listings, hosts, locations, prices, and other details. 
</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:8.208e-3,&quot;end_time&quot;:&quot;2023-02-24T02:31:31.967159&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:31.958951&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

-   [Data](#section-one)
-   [Preprocessing](#section-two)
-   [Summary Statistics and Visualization](#section-three)
    -   [See the distribution of room types in each neighborhood
        group](#subsection-one)
    -   [Description of Airbnb by Neighborhood group](#subsection-two)
    -   [Description of Airbnb by Reviews](#subsection-three)
    -   [Description of Airbnb by Prices](#subsection-four)
-   [Findings](#section-four)
-   [Conclusion](#section-five)

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:8.527e-3,&quot;end_time&quot;:&quot;2023-02-24T02:31:31.984168&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:31.975641&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="section-one"></a>

# Data

</div>

<div class="cell code" execution_count="1"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:32.003777Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:32.002960Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:33.767625Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:33.766369Z&quot;}"
papermill="{&quot;duration&quot;:1.777386,&quot;end_time&quot;:&quot;2023-02-24T02:31:33.770173&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:31.992787&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
import re
from wordcloud import WordCloud
```

</div>

<div class="cell code" execution_count="2"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:33.789267Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:33.788896Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:34.913403Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:34.912455Z&quot;}"
papermill="{&quot;duration&quot;:1.136677,&quot;end_time&quot;:&quot;2023-02-24T02:31:34.915589&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:33.778912&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df = pd.read_csv('/kaggle/input/airbnbopendata/Airbnb_Open_Data.csv', low_memory=False)
```

</div>

<div class="cell code" execution_count="3"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:34.934028Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:34.933632Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:34.942733Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:34.941532Z&quot;}"
papermill="{&quot;duration&quot;:2.11e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:34.945145&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:34.924045&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df.shape
```

<div class="output execute_result" execution_count="3">

    (102599, 26)

</div>

</div>

<div class="cell code" execution_count="4"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:34.964426Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:34.963168Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:34.969728Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:34.968808Z&quot;}"
papermill="{&quot;duration&quot;:1.8233e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:34.972007&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:34.953774&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df.columns
```

<div class="output execute_result" execution_count="4">

    Index(['id', 'NAME', 'host id', 'host_identity_verified', 'host name',
           'neighbourhood group', 'neighbourhood', 'lat', 'long', 'country',
           'country code', 'instant_bookable', 'cancellation_policy', 'room type',
           'Construction year', 'price', 'service fee', 'minimum nights',
           'number of reviews', 'last review', 'reviews per month',
           'review rate number', 'calculated host listings count',
           'availability 365', 'house_rules', 'license'],
          dtype='object')

</div>

</div>

<div class="cell code" execution_count="5"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:34.991236Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:34.990865Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:35.077271Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:35.075841Z&quot;}"
papermill="{&quot;duration&quot;:9.8495e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:35.079297&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:34.980802&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 102599 entries, 0 to 102598
    Data columns (total 26 columns):
     #   Column                          Non-Null Count   Dtype  
    ---  ------                          --------------   -----  
     0   id                              102599 non-null  int64  
     1   NAME                            102349 non-null  object 
     2   host id                         102599 non-null  int64  
     3   host_identity_verified          102310 non-null  object 
     4   host name                       102193 non-null  object 
     5   neighbourhood group             102570 non-null  object 
     6   neighbourhood                   102583 non-null  object 
     7   lat                             102591 non-null  float64
     8   long                            102591 non-null  float64
     9   country                         102067 non-null  object 
     10  country code                    102468 non-null  object 
     11  instant_bookable                102494 non-null  object 
     12  cancellation_policy             102523 non-null  object 
     13  room type                       102599 non-null  object 
     14  Construction year               102385 non-null  float64
     15  price                           102352 non-null  object 
     16  service fee                     102326 non-null  object 
     17  minimum nights                  102190 non-null  float64
     18  number of reviews               102416 non-null  float64
     19  last review                     86706 non-null   object 
     20  reviews per month               86720 non-null   float64
     21  review rate number              102273 non-null  float64
     22  calculated host listings count  102280 non-null  float64
     23  availability 365                102151 non-null  float64
     24  house_rules                     50468 non-null   object 
     25  license                         2 non-null       object 
    dtypes: float64(9), int64(2), object(15)
    memory usage: 20.4+ MB

</div>

</div>

<div class="cell code" execution_count="6"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:35.098525Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:35.098173Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:35.183924Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:35.182725Z&quot;}"
papermill="{&quot;duration&quot;:9.8379e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:35.186430&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:35.088051&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df.describe()
```

<div class="output execute_result" execution_count="6">

                     id       host id            lat           long  \
    count  1.025990e+05  1.025990e+05  102591.000000  102591.000000   
    mean   2.914623e+07  4.925411e+10      40.728094     -73.949644   
    std    1.625751e+07  2.853900e+10       0.055857       0.049521   
    min    1.001254e+06  1.236005e+08      40.499790     -74.249840   
    25%    1.508581e+07  2.458333e+10      40.688740     -73.982580   
    50%    2.913660e+07  4.911774e+10      40.722290     -73.954440   
    75%    4.320120e+07  7.399650e+10      40.762760     -73.932350   
    max    5.736742e+07  9.876313e+10      40.916970     -73.705220   

           Construction year  minimum nights  number of reviews  \
    count      102385.000000   102190.000000      102416.000000   
    mean         2012.487464        8.135845          27.483743   
    std             5.765556       30.553781          49.508954   
    min          2003.000000    -1223.000000           0.000000   
    25%          2007.000000        2.000000           1.000000   
    50%          2012.000000        3.000000           7.000000   
    75%          2017.000000        5.000000          30.000000   
    max          2022.000000     5645.000000        1024.000000   

           reviews per month  review rate number  calculated host listings count  \
    count       86720.000000       102273.000000                   102280.000000   
    mean            1.374022            3.279106                        7.936605   
    std             1.746621            1.284657                       32.218780   
    min             0.010000            1.000000                        1.000000   
    25%             0.220000            2.000000                        1.000000   
    50%             0.740000            3.000000                        1.000000   
    75%             2.000000            4.000000                        2.000000   
    max            90.000000            5.000000                      332.000000   

           availability 365  
    count     102151.000000  
    mean         141.133254  
    std          135.435024  
    min          -10.000000  
    25%            3.000000  
    50%           96.000000  
    75%          269.000000  
    max         3677.000000  

</div>

</div>

<div class="cell code" execution_count="7"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:35.206513Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:35.205618Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:35.273208Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:35.272131Z&quot;}"
papermill="{&quot;duration&quot;:7.9709e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:35.275187&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:35.195478&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df.isnull().sum()
```

<div class="output execute_result" execution_count="7">

    id                                     0
    NAME                                 250
    host id                                0
    host_identity_verified               289
    host name                            406
    neighbourhood group                   29
    neighbourhood                         16
    lat                                    8
    long                                   8
    country                              532
    country code                         131
    instant_bookable                     105
    cancellation_policy                   76
    room type                              0
    Construction year                    214
    price                                247
    service fee                          273
    minimum nights                       409
    number of reviews                    183
    last review                        15893
    reviews per month                  15879
    review rate number                   326
    calculated host listings count       319
    availability 365                     448
    house_rules                        52131
    license                           102597
    dtype: int64

</div>

</div>

<div class="cell code" execution_count="8"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:35.295186Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:35.294809Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:35.954259Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:35.952990Z&quot;}"
papermill="{&quot;duration&quot;:0.671626,&quot;end_time&quot;:&quot;2023-02-24T02:31:35.956331&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:35.284705&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
corr = df.corr()
f, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
```

<div class="output execute_result" execution_count="8">

    <AxesSubplot:>

</div>

<div class="output display_data">

![](f641c9d637833b7c004d79297a6adb1ac63e7108.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.0088e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:35.977054&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:35.966966&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

*It is reasonable the the number of reviews and reviews per month are
correlated*

</div>

<div class="cell code" execution_count="9"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:35.999228Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:35.998846Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:36.202115Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:36.201090Z&quot;}"
papermill="{&quot;duration&quot;:0.217093,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.204314&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:35.987221&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# loop through each column in the dataframe
for col in df.columns:
    # if the column has less than 50 unique values, print out the table of unique values
    if df[col].nunique() < 50:
        print(f'\n{col}:')
        display(pd.DataFrame(df[col].value_counts()))
```

<div class="output stream stdout">


    host_identity_verified:

</div>

<div class="output display_data">

                 host_identity_verified
    unconfirmed                   51200
    verified                      51110

</div>

<div class="output stream stdout">


    neighbourhood group:

</div>

<div class="output display_data">

                   neighbourhood group
    Manhattan                    43792
    Brooklyn                     41842
    Queens                       13267
    Bronx                         2712
    Staten Island                  955
    brookln                          1
    manhatan                         1

</div>

<div class="output stream stdout">


    country:

</div>

<div class="output display_data">

                   country
    United States   102067

</div>

<div class="output stream stdout">


    country code:

</div>

<div class="output display_data">

        country code
    US        102468

</div>

<div class="output stream stdout">


    instant_bookable:

</div>

<div class="output display_data">

           instant_bookable
    False             51474
    True              51020

</div>

<div class="output stream stdout">


    cancellation_policy:

</div>

<div class="output display_data">

              cancellation_policy
    moderate                34343
    strict                  34106
    flexible                34074

</div>

<div class="output stream stdout">


    room type:

</div>

<div class="output display_data">

                     room type
    Entire home/apt      53701
    Private room         46556
    Shared room           2226
    Hotel room             116

</div>

<div class="output stream stdout">


    Construction year:

</div>

<div class="output display_data">

            Construction year
    2014.0               5243
    2008.0               5225
    2006.0               5223
    2019.0               5201
    2009.0               5166
    2020.0               5158
    2010.0               5155
    2022.0               5134
    2005.0               5132
    2012.0               5131
    2003.0               5125
    2007.0               5106
    2015.0               5094
    2017.0               5066
    2011.0               5058
    2018.0               5057
    2021.0               5039
    2004.0               5037
    2013.0               5018
    2016.0               5017

</div>

<div class="output stream stdout">


    review rate number:

</div>

<div class="output display_data">

         review rate number
    5.0               23369
    4.0               23329
    3.0               23265
    2.0               23098
    1.0                9212

</div>

<div class="output stream stdout">


    license:

</div>

<div class="output display_data">

              license
    41662/AL        2

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.2191e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.230243&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.218052&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="section-two"></a>

# Preprocessing

</div>

<div class="cell code" execution_count="10"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:36.256648Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:36.256274Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:36.306156Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:36.305458Z&quot;}"
papermill="{&quot;duration&quot;:6.5527e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.308265&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.242738&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df.dropna(subset=['price', 'neighbourhood group', 'service fee'], inplace=True)
```

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.2322e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.333305&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.320983&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

Drop unnecessary variables

</div>

<div class="cell code" execution_count="11"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:36.360111Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:36.359254Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:36.383882Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:36.382593Z&quot;}"
papermill="{&quot;duration&quot;:4.0734e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.386513&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.345779&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df.drop(['id', 'host id', 'country code', 'license'], axis=1, inplace=True)
```

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.267e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.412444&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.399774&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

There were typos in the data set where Manhattan was spelled as manhatan
and Brooklyn was spelled as brookln. Let's correct this mistake.

</div>

<div class="cell code" execution_count="12"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:36.439817Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:36.439479Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:36.455541Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:36.454643Z&quot;}"
papermill="{&quot;duration&quot;:3.2433e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.457805&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.425372&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df['neighbourhood group'] = df['neighbourhood group'].replace({'manhatan': 'Manhattan','brookln': 'Brooklyn'})
```

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.2893e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.483916&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.471023&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

Prices were listed with dollar signs. Let's change its format.

</div>

<div class="cell code" execution_count="13"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:36.511250Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:36.510382Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:36.686887Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:36.686153Z&quot;}"
papermill="{&quot;duration&quot;:0.192398,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.689043&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.496645&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df['price'] = df['price'].apply(lambda x: re.sub(r'[^\d.]+', '', x)).astype(float)
```

</div>

<div class="cell code" execution_count="14"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:36.716508Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:36.715967Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:36.890445Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:36.889155Z&quot;}"
papermill="{&quot;duration&quot;:0.191401,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.893036&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.701635&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
df['service fee'] = df['service fee'].apply(lambda x: re.sub(r'[^\d.]+', '', x)).astype(float)
```

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.3694e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.919711&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.906017&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

What can we learn about different hosts and areas?

What can we learn from predictions? (ex: locations, prices, reviews,
etc)

Which hosts are the busiest and why?

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.2282e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.945947&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.933665&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="section-three"></a>

# Summary Statistics and Visualizations:

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.2369e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:36.971034&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.958665&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="subsection-one"></a>

## Description of Airbnb by Neighborhood Group

</div>

<div class="cell code" execution_count="15"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:36.998233Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:36.997909Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:37.131015Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:37.130227Z&quot;}"
papermill="{&quot;duration&quot;:0.149092,&quot;end_time&quot;:&quot;2023-02-24T02:31:37.133002&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:36.983910&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# Get the data to plot
grouped_data = df.groupby('neighbourhood group')['instant_bookable'].count().sort_values()

# Define colors
colors = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494']

# Create the bar chart
fig, ax = plt.subplots()
ax.barh(grouped_data.index, grouped_data.values, color=colors)

# Add the data labels
for i, v in enumerate(grouped_data.values):
    ax.text(v + 50, i, str(v), color='black', fontsize=12, va='center')

# Set the chart title
plt.title('Number of Listings by Neighborhood Group')

# Show the plot
plt.show()
```

<div class="output display_data">

![](7598f3861cbd75e52d5856694eb00993445db429.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.6636e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:37.167137&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:37.150501&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
    💬, &nbsp;  We see that Manhattan has to most number of listings.
</div>

</div>

<div class="cell code" execution_count="16"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:37.202664Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:37.202279Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:37.368444Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:37.367602Z&quot;}"
papermill="{&quot;duration&quot;:0.186078,&quot;end_time&quot;:&quot;2023-02-24T02:31:37.370368&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:37.184290&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# Calculate the average availability of listings in each neighborhood group
avg_availability = df.groupby('neighbourhood group')['availability 365'].mean()

#set the colors
color = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494', '#253494']
# Create a bar chart of the average availability
plt.bar(avg_availability.index, avg_availability.values, color = color)

# Set the chart title and labels
plt.title('Average Availability of Listings by Neighborhood Group')
plt.xlabel('Neighborhood Group')
plt.ylabel('Availability (Days)')

# Display the chart
plt.show()
```

<div class="output display_data">

![](e8c0dd3ea40f49626e192f0d9b86bb861bf123d1.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.3601e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:37.398778&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:37.385177&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
    💬, &nbsp; On average, we can see that Staten Island has the most number of available listings though it has the least amount of listings while Brooklyn typically has the least amount of available listings though it has the second highest listings.
</div>

</div>

<div class="cell code" execution_count="17"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:37.427999Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:37.427344Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:37.557163Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:37.556215Z&quot;}"
papermill="{&quot;duration&quot;:0.147034,&quot;end_time&quot;:&quot;2023-02-24T02:31:37.559414&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:37.412380&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# Get the data to plot
grouped_data = df.groupby('room type')['room type'].count()

# Set the colors
colors = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494']

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(grouped_data.values, labels=grouped_data.index, autopct='%1.1f%%', colors = colors, textprops={'fontsize': 14})

# Set the chart title
plt.title('Proportions of Room Types')

# Show the plot
plt.show()
```

<div class="output display_data">

![](ed963375c64269e715ae80702c4cb2760ab7d65a.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.9805e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:37.598163&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:37.578358&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
    💬, &nbsp; We can see that while there is not many hotel room type of rooms, private rooms and an entire house/apartment take are the most common types of rooms.
</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.7841e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:37.634991&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:37.617150&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="subsection-three"></a>

## Description of Aribnb by reviews

</div>

<div class="cell code" execution_count="18"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:37.663774Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:37.663462Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:38.159581Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:38.158231Z&quot;}"
papermill="{&quot;duration&quot;:0.51389,&quot;end_time&quot;:&quot;2023-02-24T02:31:38.162669&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:37.648779&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# Select the top 50 listings by number of reviews per month
top_50 = df.nlargest(50, 'reviews per month')

# Sort the data by number of reviews per month in descending order
top_50 = top_50.sort_values('reviews per month', ascending=False)

# Define the color map to use
cmap = plt.get_cmap('GnBu')

# Create the bar chart
fig, ax = plt.subplots(figsize=(12,6))
bars = ax.bar(top_50['NAME'], top_50['reviews per month'])

# Set the color of each bar based on its value
for i in range(len(bars)):
    bars[i].set_color(cmap((bars[i].get_height() - np.min(top_50['reviews per month'])) 
                           / (np.max(top_50['reviews per month']) - np.min(top_50['reviews per month']))))

# Set the chart title and axis labels
plt.title('Top 50 Listings by Number of Reviews per Month', fontsize=16)
plt.xlabel('Listing Name', fontsize=14)
plt.ylabel('Number of Reviews per Month', fontsize=14)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, fontsize=12)

# Show the plot
plt.show()
```

<div class="output display_data">

![](f5e70857f23cfe62f54fcc15d040f237329c63d7.png)

</div>

</div>

<div class="cell code" execution_count="19"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:38.198645Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:38.198305Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:38.521037Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:38.520125Z&quot;}"
papermill="{&quot;duration&quot;:0.343778,&quot;end_time&quot;:&quot;2023-02-24T02:31:38.523678&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:38.179900&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# Calculate the average review rate number for each combination of Construction year and neighborhood
df_pivot = pd.pivot_table(df, values='review rate number', index='Construction year', columns='neighbourhood group')

# Create a heatmap
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(df_pivot, cmap='GnBu')

# Set the chart title and axis labels
plt.title('Average Review Rate Number by Construction Year and Neighborhood', fontsize=16)
plt.xlabel('Neighborhood', fontsize=14)
plt.ylabel('Construction Year', fontsize=14)

# Show the plot
plt.show()
```

<div class="output display_data">

![](ef3885892f5b7ecb3db67154088afb5ee7d32621.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.8479e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:38.561582&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:38.543103&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="subsection-four"></a>

## Description of Airbnb by Prices

</div>

<div class="cell code" execution_count="20"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:38.600848Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:38.600522Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:38.947728Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:38.947119Z&quot;}"
papermill="{&quot;duration&quot;:0.369384,&quot;end_time&quot;:&quot;2023-02-24T02:31:38.949513&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:38.580129&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
sns.set(style="ticks")

# Define the color palette
palette = sns.color_palette("GnBu", 5)

# Create the box plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="room type", y="price", data=df, palette=palette)

# Set the chart title and axis labels
ax.set_title('Distribution of Prices by Room Type', fontsize=16)
ax.set_xlabel('Room Type', fontsize=14)
ax.set_ylabel('Price', fontsize=14)

# Show the plot
plt.show()
```

<div class="output display_data">

![](c3c90532803674bdfa28fd8bde0f7388e023f8d2.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.9464e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:38.988341&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:38.968877&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
    💬, &nbsp; We can see there is not much difference in prices for each types of rooms.
</div>

</div>

<div class="cell code" execution_count="21"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:39.028615Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:39.027956Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:39.348741Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:39.347999Z&quot;}"
papermill="{&quot;duration&quot;:0.343096,&quot;end_time&quot;:&quot;2023-02-24T02:31:39.350553&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:39.007457&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# Create a scatter plot of price vs. service fee
plt.scatter(df['price'], df['service fee'])

# Set the x-axis and y-axis labels
plt.xlabel('Price')
plt.ylabel('Service Fee')

# Set the plot title
plt.title('Relationship between Price and Service Fee')

# Show the plot
plt.show()
```

<div class="output display_data">

![](7cfb22ade305f7cb61cefc74211e8562a6dec91f.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:1.9605e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:39.389746&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:39.370141&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
    💬, &nbsp; We should expect a service fee that correlates with price
</div>

</div>

<div class="cell code" execution_count="22"
execution="{&quot;iopub.execute_input&quot;:&quot;2023-02-24T02:31:39.431663Z&quot;,&quot;iopub.status.busy&quot;:&quot;2023-02-24T02:31:39.431032Z&quot;,&quot;iopub.status.idle&quot;:&quot;2023-02-24T02:31:45.123898Z&quot;,&quot;shell.execute_reply&quot;:&quot;2023-02-24T02:31:45.123041Z&quot;}"
papermill="{&quot;duration&quot;:5.719852,&quot;end_time&quot;:&quot;2023-02-24T02:31:45.129601&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:39.409749&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

``` python
# Combine all the house rules into a single string
text = ' '.join(list(df['house_rules'].dropna()))

# Create the WordCloud object
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

<div class="output display_data">

![](40b625f626bde07815f1fbb74745f17a00491a9c.png)

</div>

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:2.4795e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:45.179917&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:45.155122&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="section-four"></a>

# 📚 Findings

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:2.4615e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:45.229059&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:45.204444&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

Our analysis shows that, on average, Staten Island has the highest
number of available listings despite having the lowest total number of
listings. Meanwhile, Brooklyn typically has the lowest number of
available listings despite having the second-highest total number of
listings. The majority of the rooms available are private or entire
home/apartment. Staten Island typically has a higher average review
rate, while Brooklyn and Manhattan have lower average review rates.
Prices do not appear to vary with room type, and price should be a good
indicator of service fees. The "word map" shows that "smoking," "pet,"
"rules," "don't," and "respectful/respect" are the most frequently
mentioned words in the rules made by hosts.

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:2.4363e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:45.278258&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:45.253895&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

<a id="section-five"></a>

# 😃 Conclusion

</div>

<div class="cell markdown"
papermill="{&quot;duration&quot;:2.4259e-2,&quot;end_time&quot;:&quot;2023-02-24T02:31:45.327059&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2023-02-24T02:31:45.302800&quot;,&quot;status&quot;:&quot;completed&quot;}"
tags="[]">

Airbnb has become an important part of the hospitality industry,
especially in big cities like New York. Our data analysis project
provides valuable insights into the Airbnb market in New York City. The
findings can be useful for travelers who want to find the best deals and
hosts who want to improve their listings.

</div>
