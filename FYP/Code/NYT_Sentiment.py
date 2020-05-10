from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()

analyser = SentimentIntensityAnalyzer()
stop_words=set(stopwords.words("english"))
ps = PorterStemmer()

named_entities = []
entities_to_remove = []

def remove_entities(article):
    for x in entities_to_remove:
        for entity in x:
            print(entity)
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
            elif ' '+entity+")" in article: 
                article = article.replace(' '+entity,'')
            elif ' '+entity+"?" in article:
                article = article.replace(' '+entity,'')
            elif ' '+entity+"!" in article: 
                article = article.replace(' '+entity,'')
            elif '"'+entity+' ' in article:
                article = article.replace(entity+' ','')
            elif "'"+entity+' ' in article:
                article = article.replace(entity+' ','')
            elif "["+entity+' ' in article:
                article = article.replace(entity+' ','')
            elif "("+entity+' ' in article: 
                article = article.replace(entity+' ','')
            elif "["+entity+']' in article:
                article = article.replace(entity,'')
            elif "("+entity+')' in article: 
                article = article.replace(entity,'')
            elif "'"+entity+"'" in article:
                article = article.replace(entity,'')
            elif '"'+entity+'"' in article:
                article = article.replace(entity,'')
    return(article)
        
def get_continuous_chunks(named_entities,text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
#    prev = None
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

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score

# The ``compound`` score is computed by summing the valence scores 
# of each word in the lexicon, adjusted according to the rules, 
# and then normalized to be between -1 (most extreme negative) and +1 
# (most extreme positive). This is the most useful metric if you want 
# a single unidimensional measure of sentiment for a given sentence. 
# Calling it a 'normalized, weighted composite score' is accurate.

def get_sentiment_score(data):
    data['combined'] =  data['body_text'].astype(str) + data['body_text_overflow1'].astype(str) + data['body_text_overflow2'].astype(str) + data['body_text_overflow3'].astype(str)
    articles = data['combined']
    print(articles.head(5))
    scores = []
    timer = 0 # delete later
    for article in articles:
        timer += 1
        print("Working on article: ", timer)
        if(pd.isnull(article) == False):
            line = []
            score = sentiment_preproc(article)
            score = str(score)
            neg = score.split(",",1)[0][8:]
            line.append(neg)
            neu = score.split(",",2)[1][7:]
            line.append(neu)
            pos = score.split(",",3)[2][7:]
            line.append(pos)
            com = score.split(",",4)[3][13:]
            com = com.replace("}","")
            line.append(com)
            scores.append(line)
    scores_df = pd.DataFrame(scores, columns=['negative_score','neutral_score','positive_score','compund_score'])
    return scores_df

def sentiment_preproc(t):
    get_continuous_chunks(named_entities,t)
    entities_to_remove = Counter(named_entities)
    entities_to_remove = sorted(entities_to_remove)
    t = remove_entities(t)

    score = analyser.polarity_scores(t)
    
    #tokenized_text = sent_tokenize(t)
    tokenized_word = word_tokenize(t)
    
    filtered_sent=[]
    tokenized_sent = tokenized_word
    for w in tokenized_sent:
        if w not in stop_words:
            filtered_sent.append(w)
    
    lemmatize_words = []
    for w in filtered_sent:
        lemmatize_words.append(lemma.lemmatize(w))
    thing = ' '.join(lemmatize_words)
    score = analyser.polarity_scores(thing)
    return score