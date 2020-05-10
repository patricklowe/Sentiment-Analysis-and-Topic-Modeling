from yanytapi import SearchAPI
import pandas as pd
from NYT_PreProcessing import *
from NYT_Sentiment import *
from NYT_TopicModelling import *

#   Store my API key for access to NYT
api = SearchAPI("0qun6D4s4ikYGONQAxMysKGyt6Coha1F")

#   Set search parameters, dates in format YYYY-MM-DD
keys = ["mental health"]
starts = ["20190101"]#"20100101","20110101","20120101","20130101","20140101","20150101","20160101","20170101","20180101","20190101"]
end = ["20190201"]#"20101201","20111201","20121201","20131201","20141201","20151201","20161201","20171201","20181201","20191201"]

start(starts,end,keys,api)
#   Use a webscraper to get full body text (not provided by API)
body_text = pd.Series()
for date in starts:
    raw_data = pd.read_csv( date + '.csv' )
    print("Looking into " + date)
    
    body_text_df = pd.DataFrame()
    body_text_df = get_text(raw_data)
    
    #   Add a new column of the articles body of text, export
    combined_data = raw_data.join(body_text_df)
    export_df(combined_data,"kept_data"+date)
    #   Use 'byline' to gather names of authors
    people_df = pd.DataFrame()
    people_df = get_person(raw_data['byline'])
    preproc_data = combine_df(combined_data,people_df,"authors")
    preproc_data2 = preproc_data.drop(['byline'], axis=1) 
    preproc_data3 = preproc_data2[preproc_data2['authors'].notna()]
    
    #   Split authors, up to 4
    preproc_data4 = preproc_data3.reset_index(drop=True)
    authors = split_authors(preproc_data4['authors'])
    preproc_data5 = pd.concat([preproc_data4, authors], axis=1)
    preproc_data6 = preproc_data5.drop(['authors'], axis=1)
    
    #   Use the 'headline' to get the kicker, other data is obsolete
    kicker_df = pd.DataFrame()
    kicker_df = get_kicker(raw_data['headline'])
    preproc_data7 = combine_df(preproc_data6,kicker_df,"kicker")
    preproc_data8 = preproc_data7.drop(['headline'], axis=1)    
    export_df(preproc_data8,"kept_data2"+date)
    
    #   Apply Sentiment Analysis
    SA_Score_df = get_sentiment_score(preproc_data8)     
    preproc_data9 = pd.concat([preproc_data8, SA_Score_df], axis=1, sort=False)
    text = preproc_data9['combined']
    cases = []
    for entry in text:
        if( pd.isnull(entry) == False):
            print(entry)
            cases.append( entry.lower().count("mental") )
    cases_df = pd.DataFrame(cases, columns=['cases'])
    preproc_data9['cases'] = cases_df
    preproc_data10 = preproc_data9.drop(['combined'], axis=1)    
    preproc_data10 = drop_dups(preproc_data10)
    export_df(preproc_data10,"kept_data3"+date)
    
# Apply Topic Modelling
for date in starts:
    modelling_df = pd.read_csv("kept_data3"+ date +'.csv' )
    modelling_df = drop_dups(modelling_df)
    export_df(modelling_df ,"kept_data" + date)

merge_files()
# Apply Topic Modeling

TM_Main()
# Get Dates
topics = pd.read_csv('applied.csv')

full = pd.read_csv('All_Data.csv')
topics = topics.drop(['article_text','article_text_nostop_extra'], axis = 1)
result = pd.concat([full, topics], axis=1)    
result = full
dates = get_dates(result)
result = pd.concat([result, dates], axis=1)    
export_df(result,"OVERALL")