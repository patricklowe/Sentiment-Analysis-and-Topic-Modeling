#import os
#import sys
#import json
#import datetime
import pickle
#import gc
#import time
from operator import itemgetter
from collections import defaultdict
import pandas as pd
import numpy as np
#import gensim
from gensim import matutils
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from collections import Counter
import operator
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamodel import LdaModel
#from nltk.corpus import stopwords
import pprint
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
pp = pprint.PrettyPrinter(indent=4)
#import logging

def get_continuous_chunks(named_entities,text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    named_entities += continuous_chunk


def remove_entities(article,entities_to_remove):
    for entity in entities_to_remove:
        if ' '+entity+' ' in article:
            article = article.replace(entity+' ','') 
        elif ' '+entity+'.' in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+',' in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+':' in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+'-' in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+';' in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+'"' in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+"'" in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+"]" in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+")" in article: # added later
            article = article.replace(' '+entity,'')
        elif ' '+entity+"?" in article:
            article = article.replace(' '+entity,'')
        elif ' '+entity+"!" in article: # added later
            article = article.replace(' '+entity,'')
        elif '"'+entity+' ' in article:
            article = article.replace(entity+' ','')
        elif "'"+entity+' ' in article:
            article = article.replace(entity+' ','')
        elif "["+entity+' ' in article:
            article = article.replace(entity+' ','')
        elif "("+entity+' ' in article: # added later
            article = article.replace(entity+' ','')
        elif "["+entity+']' in article:
            article = article.replace(entity,'')
        elif "("+entity+')' in article: # added later
            article = article.replace(entity,'')
        elif "'"+entity+"'" in article:
            article = article.replace(entity,'')
        elif '"'+entity+'"' in article:
            article = article.replace(entity,'')
    return(article)
    
def cross_val_topics_50p(no_of_topics,corpus,dictionary,validation_50p):
    model = LdaModel(corpus, num_topics=no_of_topics, id2word = dictionary, passes=50, alpha='auto', eval_every=2000)
    globals()['ldamodel_%dt_50p_autoalpha_val' % no_of_topics] = model
    model.save('lda_{}t_50p_autoalpha_val.model'.format(no_of_topics))
    validation_50p[no_of_topics] = model.print_topics(num_topics=-1, num_words=50)
    
def get_art_topics_val(val_models_to_check,corpus):
    model_num = 15 # revert to 12?
    for model in val_models_to_check:
        global art_topics_val
        art_topics_val = []
        for article in corpus:
            art_topics_val.append(model.get_document_topics(article, 
                                            minimum_probability=0.15,   
                                           minimum_phi_value=None, 
                                           per_word_topics=False)
             )
        for i in art_topics_val:
            i.sort(key=itemgetter(1),reverse=True)
        globals()['art_topics_ldamodel_%dt_50p_autoalpha_val' % model_num] = art_topics_val
        with open('art_topics_lda_{}t_50p_autoalpha_val.pickle'.format(model_num),'wb') as file:
            pickle.dump(art_topics_val,file)
        model_num += 1
        
def get_av_prob_scores(applied):
    applied_score = [[x[1] for x in article] for article in applied]
    applied_score = [np.mean(article) for article in applied_score]
    applied_score = np.mean(applied_score)
    return applied_score
    
def TM_Main():
    print("Loading Step 1...")
    modelling_df = pd.read_csv( "All_Data.csv" ) 
    #modelling_df = pd.read_csv( "kept_data320100101.csv" ) 
    print(modelling_df.head())
    print(modelling_df.size)
    modelling_df = modelling_df.replace(np.nan,"",regex=True)
    preproc_data = pd.DataFrame()
    preproc_data['body_text'] = modelling_df['body_text'] + " " + modelling_df['body_text_overflow1'] + " " + modelling_df['body_text_overflow2'] + " " + modelling_df['body_text_overflow3']
    
    # Get list of all named entities in articles
    named_entities = []
    article = 0
    
    print("Getting named entities")
    
    for a in preproc_data['body_text']:
        get_continuous_chunks(named_entities,a)
        article = a
    
    named_entities_counts = Counter(named_entities)
    named_entities_counts = sorted(named_entities_counts.items(), key=operator.itemgetter(1),reverse=True)
    common_entities = []
    for i in range(0,len(named_entities_counts)):
        common_entities.append(named_entities_counts[i][0])
    entities_to_remove = sorted(common_entities)
    
    print("Removing Entities")
    
    preproc_data['body_text_noent'] = [remove_entities(x,entities_to_remove) for x in preproc_data['body_text']]
    preproc_data.to_csv('kept_data.csv', index=False)

    print("Loading Step 2...")
    print("Tokenizing")
    #   TOKENIZE
    preproc_data['tokenized_text'] = [word_tokenize(x) for x in preproc_data['body_text_noent']]
    #   Remove Punctuation
    preproc_data['tokenized_nopunc'] = [[word for word in x if word.isalpha()] for x in preproc_data['tokenized_text']]
    #   Remove Capitalisation
    preproc_data['tokenized_nopunc_lower'] = [[word.lower() for word in x] for x in preproc_data['tokenized_nopunc']]
    #   Remove Stopwords
    extra_stop_words = ['big','small','low','high','none','may','among','within','don','t','day','etc','around','frequent','including','even','can','likely','will','like','today','bit','put','aim','s','got','really','huge','see','almost','already','much','recent',   'many','change',    'changes',       'someone','said','says','gives','give',
    'people','new','say','least','first','last','second','one','two','go','goes','take','going','taking','just','can''cannot','keep','keeps','also','done','good','get','without','told','might','time','unable',  'able',  'know','end','now','want','didn','back','doesn','couldn','since','shouldn','seen','works','zero','every','each','other','ever','neither','ll','mr','ms','mrs','think','tomorrow','way','still','know','later','fine','let','went','night','ve','must','act',  're','c','b', 'a','done','began','ones','m','soon','word','along','main','q','lot','e', 'd','entire','year','mean','means','important','always','something','rather','either','makes','make','uses','use','enough','w','d','never','giving','o','involve','involes','involving','little','inside','sat','third','fourth','fifth','sixth','next','given','million','billion','millions','billions','option','options','full','complete','need','needs','set','manage','sets','manages','bring','brings','brought','try','tries','tried''week','former','monday','tuesday','wednesday','thursday','friday','saturday','sunday','spent','spend', 'spends','month','months','send','sends','sent','went','january','february','march','april','may','june','july','august','september','october','november','december','allow','process',
    'old','times','nearly','looking','looks','look','thinly','becoming','stay','stays','took','takes','take','types', 'type','thought', 'though','idea','clear','clearly','behind','half','us','less','claim','claims','long', 'short','smaller','larger','bigger','largest','biggest','smallest','longer','shorter','short','long','extreme','severe','largely','anymore','years','spoke','give','gave','given','gives','reportedly','supposedly','alledgedly','please','received','receive','receives','longtime','best','existing','putting','put','puts','whose','yesterday','thing','week','another','month','day','come']
    preproc_data['tokenized_nopunc_lower'] = [[word for word in x if not word in extra_stop_words] for x in preproc_data['tokenized_nopunc_lower']]
    
    print("Lemmatizing")
    #   Lemmatize
    preproc_data['tokenized_nopunc_lower_nostop_extra_lemmatized'] = [[lemma.lemmatize(word) for word in x] for x in preproc_data['tokenized_nopunc_lower']]
    
    # List of Extra Stop words
    stopwords = ['boyd','rev','wu','did','durst','week','another','thing','month','day','come',
        'york','away','left','wrote','came','tell','asked',
        'left','right','hand','point','often','talk','head','point','ago','whether',
        'hour','group','became','become','becomes','often','sometimes','usually','ha','wa']
    
    print("Removing Stopwords")
    art = pd.DataFrame()
    art['article_text'] = preproc_data['tokenized_nopunc_lower_nostop_extra_lemmatized']
    art['article_text_nostop_extra'] = [[word for word in x if not word in stopwords] for x in art['article_text']]
    art['string'] = [' '.join(x) for x in art['article_text_nostop_extra']]
    
    #   MAY NEED TO CHANGE 27 higher or lowe depending on # doc in corpus
    #       0.95 * num_doc > min_df
    #       Currently: 0.95 * 29 = 27.55 > 27
    
    print("Vectorizing")
    vec = CountVectorizer(max_df =.95,min_df = 3,stop_words='english')
    counts = vec.fit_transform(art['string']).transpose()
    corpus = matutils.Sparse2Corpus(counts)
    dictionary = dict((v, k) for k, v in vec.vocabulary_.items())
    
    validation_50p = {}
    print("Loading Step 3...")
    for i in np.arange(1,21):
        print('Running LDA with 50 passes for {} topics...'.format(i))
        cross_val_topics_50p(i,corpus,dictionary,validation_50p)
    
    pp.pprint(validation_50p)
    
    val_models_to_check = [ldamodel_15t_50p_autoalpha_val]
    get_art_topics_val(val_models_to_check,corpus)
    applied_topics = [art_topics_ldamodel_15t_50p_autoalpha_val]
    scores_200p = defaultdict()
    mod_num = 15 # revert to 12?
    for applied in applied_topics:
        scores_200p[mod_num] = get_av_prob_scores(applied)
        mod_num += 1
        
    print("Loading Step 4...")
    ldamodel15 = LdaModel(corpus, num_topics=15, id2word = dictionary, passes=300, alpha='auto', eval_every=1000)
    pp.pprint(ldamodel15.show_topics(num_topics=-1, num_words=50, formatted=False))
    
    Healthcare = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Healthcare_words = []
    for word in np.arange(0,len(Healthcare))[::2]:
        Healthcare_words.append(int(float(Healthcare[word].split('*')[0])*1000)*[Healthcare[word].split('*')[1].split('"')[1]])
        
    Sports = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Sports_words = []
    for word in np.arange(0,len(Sports))[::2]:
        Sports_words.append(int(float(Sports[word].split('*')[0])*1000)*[Sports[word].split('*')[1].split('"')[1]])
    
    Research = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Research_words = []
    for word in np.arange(0,len(Research))[::2]:
        Research_words.append(int(float(Research[word].split('*')[0])*1000)*[Research[word].split('*')[1].split('"')[1]])
        
    Guns = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Guns_words = []
    for word in np.arange(0,len(Guns))[::2]:
        Guns_words.append(int(float(Guns[word].split('*')[0])*1000)*[Guns[word].split('*')[1].split('"')[1]])

    Books = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Books_words = []
    for word in np.arange(0,len(Books))[::2]:
        Books_words.append(int(float(Books[word].split('*')[0])*1000)*[Books[word].split('*')[1].split('"')[1]])
    
    Police = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Police_words = []
    for word in np.arange(0,len(Police))[::2]:
        Police_words.append(int(float(Police[word].split('*')[0])*1000)*[Police[word].split('*')[1].split('"')[1]])
        
    Drugs = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Drugs_words = []
    for word in np.arange(0,len(Drugs))[::2]:
        Drugs_words.append(int(float(Drugs[word].split('*')[0])*1000)*[Drugs[word].split('*')[1].split('"')[1]])

    Community = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Community_words = []
    for word in np.arange(0,len(Community ))[::2]:
        Community_words.append(int(float(Community[word].split('*')[0])*1000)*[Community[word].split('*')[1].split('"')[1]])

    Treatment = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Treatment_words = []
    for word in np.arange(0,len(Treatment))[::2]:
        Treatment_words.append(int(float(Treatment[word].split('*')[0])*1000)*[Treatment[word].split('*')[1].split('"')[1]])

    Jury = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Jury_words = []
    for word in np.arange(0,len(Jury))[::2]:
        Jury_words.append(int(float(Jury[word].split('*')[0])*1000)*[Jury[word].split('*')[1].split('"')[1]])

    Edu = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Edu_words = []
    for word in np.arange(0,len(Edu))[::2]:
        Edu_words.append(int(float(Edu[word].split('*')[0])*1000)*[Edu[word].split('*')[1].split('"')[1]])

    Politics = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Politics_words = []
    for word in np.arange(0,len(Politics))[::2]:
        Politics_words.append(int(float(Politics[word].split('*')[0])*1000)*[Politics[word].split('*')[1].split('"')[1]])

    WSHealth = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    WSHealth_words = []
    for word in np.arange(0,len(WSHealth))[::2]:
        WSHealth_words.append(int(float(WSHealth[word].split('*')[0])*1000)*[WSHealth[word].split('*')[1].split('"')[1]])
        
    Prison = ldamodel15.show_topics(num_topics=-1, num_words=50)[0][1].split()
    Prison_words = []
    for word in np.arange(0,len(Prison))[::2]:
        Prison_words.append(int(float(Prison[word].split('*')[0])*1000)*[Prison[word].split('*')[1].split('"')[1]])

    art_topics = []
    for article in corpus:
        art_topics.append(ldamodel15.get_document_topics(article, minimum_probability=0.015, minimum_phi_value=None, per_word_topics=False) )
    art['topics'] = art_topics
    art[['topics']].head()
    
    
    topics = {
            0:	'Healthcare/Insurance',
            1:	'Sports/Games',
            2:	'Research',
            3:	'State/Guns',
            4:	'Books/Friends',
            5:	'Police/Family',
            6:	'State/Drugs/Addiction',
            7:	'State/Community',
            8:	'Healthcare/Treatment',
            9:	'Judicial System',
            10:	'Education',
            11:	'State/Politics',
            12:	'Womens Sexual Health',
            13:	'Prison',
            14:	'Military'
    }
    for t in np.arange(0,14):
        art[topics[t]] = [[i[1] if i[0]==t else 0 for i in a] for a in art['topics']]
        art[topics[t]] = [[i for i in a if i > 0] for a in art[topics[t]]]
        art[topics[t]] = art[topics[t]].apply(lambda x: 0 if len(x)==0 else x[0])
    art.to_csv("applied.csv", index=False)
    return art      
