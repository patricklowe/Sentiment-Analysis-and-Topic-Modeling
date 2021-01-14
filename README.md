# Lunacy, Mental Health, and Well-Being: Visualizing the change in online media’s discussion of mental health.
## Goal
Understand an online media's discussion of mental health over a 10 year period to see if there is a change, either positive or negative.
## Outline
This project collects article data from The New York Times over a 10 year period which discuss mental health. It uses Natural Languange Processing (NLP) to performing topic modelling - categorical breakdown of discussion around mental health -
, and other features of NLP - entity removal, lemmatization, and tokenization - to achieve different structures of conversation. Then Sentiment Analysis is performed on each article to identify if it has an overall positive, neutral or negative sentiment towards mental health.
Finally the data is graphed to draw conclusions.

## Challenges
- The NYT API only allowed for meta data to be collected, the article text was collected using a paid subscription and a webscrapper to collect the full article. This data cannot be shared without a license and therefore has been removed from this repository.
- Articles were pulled if they contained the key word 'mental health', this means articles discussing the topic without that word could be missed.
- As above, articles could also be pulled mentioning the key word 'mental health' without being the focal point of discussion of said article, as such I found the average count of the keyword being mentioned and any article below that threshold was removed from the analysis
- Performing analysis on an entire article could give different results than performing sentiment analysis on key sections discussing mental health
