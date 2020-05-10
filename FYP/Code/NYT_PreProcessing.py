import os
import sys
import pandas as pd
import numpy as np
import json
import requests
import datetime
import pickle
#import nltk
import time
import re
import glob
from nltk import ne_chunk_sents, ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from collections import Counter
import operator
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import urllib.request
import string

lemma = WordNetLemmatizer()

#   Gather all articles containing Mental Health
def start(starts,end,key,api):
    for i in range(0, len(starts)):
        print("Gathering: " + starts[i])
        data_frame = get_articles(api,starts[i],end[i],key)
        filename = starts[i] + ".csv"
        data_frame.to_csv(filename, index=False)


def get_articles(api,start,finish,key):
    mental_health = api.search(key, 
                           fq={"body": key,"document_type":"article"},
                           begin_date = start,
                           end_date = finish,
                           facet_field=["source", "day_of_week"],
                           facet_filter=True)

    article_segments = [[article._id,
                       article.document_type,
                       article.blog,
                       article.byline,
                       article.headline,
                       article.keywords,
                       article.lead_paragraph,
                       article.meta,
                       article.multimedia,
                       article.news_desk,
                       article.score,
                       article.section_name,
                       article.snippet,
                       article.source,
                       article.subsectoinName,
                       article.type_of_material,
                       article.uri,
                       article.web_url,
                       article.pub_date,
                       article.word_count] for article in mental_health]

    raw_data = pd.DataFrame(data=article_segments, columns=[
      "_id",
      "document_type",
      "blog",
      "byline",
      "headline",
      "keywords",
      "lead_paragraph",
      "meta",
      "multimedia",
      "news_desk",
      "score",
      "section_name",
      "snippet",
      "source",
      "subsectoinName",
      "type_of_material",
      "uri",
      "web_url",
      "pub_date",
      "word_count"])
    raw_data.to_csv('data.csv', index=False)
    return raw_data

def remove_p_tag(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def get_text(raw_data):
    row_entry1 = []
    row_entry2 = [] # for any articles which overflow character limit in excel
    row_entry3 = [] 
    row_entry4 = []
    i = 0
    for url in raw_data['web_url']:
        i += 1
        print("Getting " + str(i) + " of " + str(len(raw_data['uri'])))
        if("interactive" in url):
            print("Skipping - interactive")
            row_entry1.append("interactive")
            row_entry2.append("interactive")
            row_entry3.append("interactive")
            row_entry4.append("interactive")
        elif(".blogs." in url):
            print("Skipping - blogs")
            row_entry1.append("blog")
            row_entry2.append("blog")
            row_entry3.append("blog")
            row_entry4.append("blog")
        elif("dealbook" in url):
            print("Skipping - dealbook")
            row_entry1.append("dealbook")
            row_entry2.append("dealbook")
            row_entry3.append("dealbook")
            row_entry4.append("dealbook")
        elif("opinion" in url):
            print("OPINION")
            try: page = urllib.request.urlopen(url)
            except urllib.error.URLError as e:
                print(e.reason)
            text = ""
            page = urllib.request.urlopen(url)
            soup = BeautifulSoup(page,"html.parser")
            table = soup.find("section", attrs={"name": "articleBody"})
            response = requests.get(url)
            if( str(type(table)) != "<class 'NoneType'>"):
                results = table.find_all("div", attrs={"class": "css-1fanzo5 StoryBodyCompanionColumn"})
                for result in results:
                    data = result.find_all("p")
                    for para in data:
                        text += remove_p_tag(para.text)
                info = text[:4680]
                info2 = text[4680:9360]
                info3 = text[9360:14040]
                info4 = text[14040:720]
                row_entry1.append(info)
                row_entry2.append(info2)
                row_entry3.append(info3)
                row_entry4.append(info4)
        else:
            print("looking into - article")
            text = ""
            try: page = urllib.request.urlopen(url)
            except urllib.error.URLError as e:
                print(e.reason)
            soup = BeautifulSoup(page,"html.parser")
            table = soup.find("section", attrs={"name": "articleBody"})
            response = requests.get(url)
            if( str(type(table)) != "<class 'NoneType'>"):
                results = table.find_all("div", attrs={"class": "css-1fanzo5 StoryBodyCompanionColumn"})
                for result in results:
                    data = result.find_all("p")
                    for para in data:
                        text += para.text
                info = text[:4680]
                info2 = text[4680:9360]
                info3 = text[9360:14040]
                info4 = text[14040:720]
                row_entry1.append(info)
                row_entry2.append(info2)
                row_entry3.append(info3)
                row_entry4.append(info4)
            else:
                row_entry1.append("Problem encountered.. skipping")
                row_entry2.append("Problem encountered.. skipping")
                row_entry3.append("Problem encountered.. skipping")
                row_entry4.append("Problem encountered.. skipping")
    zippedList = list( zip( row_entry1,row_entry2,row_entry3,row_entry4 ) )
    text_df = pd.DataFrame(zippedList, columns=['body_text','body_text_overflow1','body_text_overflow2','body_text_overflow3'])
    return text_df

#   Combine CSV files
def merge_files():
    extension = 'csv'
    all_filenames = [i for i in glob.glob('kept_data3*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv("All_Data.csv", index=False)


def combine_df(df_1,df_2,name):
    df_1[name] = df_2
    return df_1

def get_person(person):
    people = []
    for pers in person:
        separator = "',"
        if(pers[13:17] == "None" ):
            name = "None"
        else:
            name = pers.split(separator, 1)[0]
            name = name[17:]
        people.append(name)
    people_df = pd.DataFrame(people, columns=['author'])
    return people_df

def split_authors(data):
    authors = []
    for auth in data:
        print(auth)
        name = auth.replace(" and",",")
        name = name.split(",")
        i = 0
        temp = []
        while(i < 4):
            if(i < len(name)):
                temp.append(name[i])
            else:
                temp.append("")
            i += 1
        authors.append(temp)
    frame = pd.DataFrame(authors,columns=['author_1','author_2','author_3','author_4'])
    return frame       

def get_kicker(headline):
    kickers = []
    for kick in headline:
        name = kick.split("'kicker':", 2)[1]
        name = name.split("'content_kicker'",1)[0]
        name = name.translate(str.maketrans('', '', string.punctuation))
        name = name.replace('’','')
        kickers.append(name)
    kicker_df = pd.DataFrame(kickers, columns=['kicker'])
    return kicker_df


#   Drop uninsightful columns
def drop_uninsightful(data_frame):
    returned_data = data_frame.drop(['document_type','blog','keywords','multimedia','pub_date','news_desk','snippet','source','type_of_material','web_url','lead_paragraph'], axis = 1)
    returned_data = returned_data[~returned_data.uri.str.contains(".blogs.")]
    returned_data = returned_data[~returned_data.uri.str.contains("dealbook")]
    return returned_data 

# Drop duplicates
def drop_dups(data_frame):
    data_frame.drop_duplicates(subset=['_id'], keep='first', inplace=True)
    return data_frame

#   Export to a new file
def export_df(data_frame, filename):
    data_frame.to_csv(filename + '.csv', index=False)

def get_dates(links_df):
    years = []
    months = []
    days = []
    for link in links_df['web_url']:
        link = link.replace("https://www.nytimes.com/","")
        link = link.split("/")
        years.append(link[0])
        months.append(link[1])
        days.append(link[2]) 
    dates = pd.DataFrame({'Year': years,'Month': months,'Day': days})
    return dates