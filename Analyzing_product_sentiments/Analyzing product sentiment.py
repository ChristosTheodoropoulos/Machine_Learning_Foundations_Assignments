
# coding: utf-8

# #Predicting sentiment from product reviews
# 
# #Fire up GraphLab Create

# In[1]:

import graphlab


# #Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[2]:

products = graphlab.SFrame('amazon_baby.gl/')


# #Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[3]:

products.head()


# #Build the word count vector for each review

# In[4]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[5]:

products.head()


# In[6]:

graphlab.canvas.set_target('ipynb')


# In[7]:

products['name'].show()


# #Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[8]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[ ]:




# In[9]:

len(giraffe_reviews)


# In[10]:

giraffe_reviews['rating'].show(view='Categorical')


# #Build a sentiment classifier

# In[11]:

products['rating'].show(view='Categorical')


# ##Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[12]:

#ignore all 3* reviews
products = products[products['rating'] != 3]


# In[13]:

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[14]:

products.head()


# ##Let's train the sentiment classifier

# In[15]:

train_data,test_data = products.random_split(.8, seed=0)


# In[16]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# #Evaluate the sentiment model

# In[17]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[113]:

sentiment_model.show(view='Evaluation')


# #Applying the learned model to understand sentiment for Giraffe

# In[20]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[21]:

giraffe_reviews.head()


# ##Sort the reviews based on the predicted sentiment and explore

# In[22]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[23]:

giraffe_reviews.head()


# ##Most positive reviews for the giraffe

# In[24]:

giraffe_reviews[0]['review']


# In[25]:

giraffe_reviews[1]['review']


# ##Show most negative reviews for giraffe

# In[26]:

giraffe_reviews[-1]['review']


# In[27]:

giraffe_reviews[-2]['review']


# In[28]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[30]:

selected_words


# In[33]:




# In[38]:

def awesome_count (a):
    count = 0
    if 'awesome' in a:
        count += 1
    return(count)


# In[43]:

def great_count (a):
    count = 0
    if 'great' in a:
        count += 1
    return(count)


# In[44]:

def fantastic_count (a):
    count = 0
    if 'fantastic' in a:
        count += 1
    return(count)


# In[45]:

def amazing_count (a):
    count = 0
    if 'amazing' in a:
        count += 1
    return(count)


# In[47]:

def love_count (a):
    count = 0
    if 'love' in a:
        count += 1
    return(count)


# In[48]:

def horrible_count (a):
    count = 0
    if 'horrible' in a:
        count += 1
    return(count)


# In[49]:

def bad_count (a):
    count = 0
    if 'bad' in a:
        count += 1
    return(count)


# In[50]:

def terrible_count (a):
    count = 0
    if 'terrible' in a:
        count += 1
    return(count)


# In[51]:

def awful_count (a):
    count = 0
    if 'awful' in a:
        count += 1
    return(count)


# In[52]:

def wow_count (a):
    count = 0
    if 'wow' in a:
        count += 1
    return(count)


# In[53]:

def hate_count (a):
    count = 0
    if 'hate' in a:
        count += 1
    return(count)


# In[39]:

products['awesome'] = products['word_count'].apply(awesome_count)


# In[54]:

products['great'] = products['word_count'].apply(great_count)


# In[55]:

products['fantastic'] = products['word_count'].apply(fantastic_count)


# In[56]:

products['amazing'] = products['word_count'].apply(amazing_count)


# In[57]:

products['love'] = products['word_count'].apply(love_count)


# In[58]:

products['horrible'] = products['word_count'].apply(horrible_count)


# In[59]:

products['bad'] = products['word_count'].apply(bad_count)


# In[60]:

products['terrible'] = products['word_count'].apply(terrible_count)


# In[61]:

products['awful'] = products['word_count'].apply(awful_count)


# In[62]:

products['wow'] = products['word_count'].apply(wow_count)


# In[71]:

products['hate'] = products['word_count'].apply(hate_count)


# In[72]:

products.head()


# In[73]:

sum_awesome = sum(products['awesome'])
sum_great = sum(products['great'])
sum_fantastic = sum(products['fantastic'])
sum_amazing = sum(products['amazing'])
sum_love = sum(products['love'])
sum_horrible = sum(products['horrible'])
sum_bad = sum(products['bad'])
sum_terrible = sum(products['terrible'])
sum_awful = sum(products['awful'])
sum_wow = sum(products['wow'])
sum_hate = sum(products['hate'])


# In[74]:

sum_awesome


# In[77]:

sum_selected_words = [sum_awesome, sum_great, sum_fantastic, sum_amazing, sum_love, sum_horrible]


# In[78]:

sum_selected_words


# In[79]:

sum_selected_words = [sum_selected_words, sum_bad]


# In[80]:

sum_selected_words


# In[81]:

sum_selected_words = sum_selected_words.append(sum_bad)


# In[83]:

sum_selected_words


# In[84]:

sum_selected_words


# In[87]:

sum_selected_words = [sum_awesome, sum_great, sum_fantastic, sum_amazing, sum_love, sum_horrible, sum_bad, sum_terrible, sum_awful, sum_wow, sum_hate]


# In[ ]:




# In[86]:

sum_selected_words


# In[88]:

train_data,test_data = products.random_split(.8, seed=0)


# In[89]:

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target ='sentiment',
                                                     features = selected_words,
                                                     validation_set = test_data)


# In[91]:

coefficients = selected_words_model['coefficients']


# In[92]:

coefficients


# In[93]:

coefficients = coefficients.sort('value', ascending=False)


# In[110]:

coefficients


# In[111]:

coefficients.print_rows(12, 5)


# In[95]:

selected_words_model.evaluate(test_data, metric='roc_curve')


# In[112]:

selected_words_model.show(view='Evaluation')


# In[97]:

champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[100]:

champ_reviews['rating'].show(view='Categorical')


# In[98]:

champ_reviews


# In[101]:

champ_reviews['predicted_sentiment'] = sentiment_model.predict(champ_reviews, output_type='probability')


# In[102]:

champ_reviews


# In[ ]:




# In[103]:

champ_reviews = champ_reviews.sort('predicted_sentiment', ascending=False)


# In[104]:

champ_reviews


# In[106]:

best_champ = selected_words_model.predict(champ_reviews[0], output_type='probability')


# In[107]:

best_champ


# In[108]:

champ_reviews[0]

