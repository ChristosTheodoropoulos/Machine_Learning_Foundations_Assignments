
# coding: utf-8

# #Document retrieval from wikipedia data
# 
# #Fire up GraphLab Create

# In[3]:

import graphlab


# #Load some text data - from wikipedia, pages on people

# In[4]:

people = graphlab.SFrame('people_wiki.gl/')


# Data contains:  link to wikipedia article, name of person, text of article.

# In[5]:

people.head()


# In[6]:

len(people)


# #Explore the dataset and checkout the text it contains
# 
# ##Exploring the entry for president Obama

# In[2]:

obama = people[people['name'] == 'Barack Obama']


# In[6]:

obama


# In[7]:

obama['text']


# ##Exploring the entry for actor George Clooney

# In[8]:

clooney = people[people['name'] == 'George Clooney']
clooney['text']


# #Get the word counts for Obama article

# In[9]:

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])


# In[10]:

print obama['word_count']


# ##Sort the word counts for the Obama article

# ###Turning dictonary of word counts into a table

# In[11]:

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])


# ###Sorting the word counts to show most common words at the top

# In[12]:

obama_word_count_table.head()


# In[13]:

obama_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# #Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[14]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[31]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
tfidf


# In[ ]:




# In[32]:

people['tfidf'] = graphlab.text_analytics.tf_idf(people['word_count'])


# ##Examine the TF-IDF for the Obama article

# In[33]:

obama = people[people['name'] == 'Barack Obama']


# In[41]:

a = obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# In[42]:

a


# In[39]:

obama


# Words with highest TF-IDF are much more informative.

# In[ ]:




# #Manually compute distances between a few people
# 
# Let's manually compare the distances between the articles for a few famous people.  

# In[43]:

clinton = people[people['name'] == 'Bill Clinton']


# In[44]:

beckham = people[people['name'] == 'David Beckham']


# ##Is Obama closer to Clinton than to Beckham?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about president Obama is closer to the one about former president Clinton than that of footballer David Beckham.

# In[45]:

graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])


# In[46]:

graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# 
# #Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  

# In[47]:

knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# #Applying the nearest-neighbors model for retrieval

# ##Who is closest to Obama?

# In[48]:

knn_model.query(obama)


# As we can see, president Obama's article is closest to the one about his vice-president Biden, and those of other politicians.  

# ##Other examples of document retrieval

# In[49]:

swift = people[people['name'] == 'Taylor Swift']


# In[50]:

knn_model.query(swift)


# In[51]:

jolie = people[people['name'] == 'Angelina Jolie']


# In[52]:

knn_model.query(jolie)


# In[53]:

arnold = people[people['name'] == 'Arnold Schwarzenegger']


# In[54]:

knn_model.query(arnold)


# In[9]:

# Answer quiz questions


# In[7]:

john = people[people['name'] == 'Elton John']


# In[8]:

john


# In[10]:

john['word_count'] = graphlab.text_analytics.count_words(john['text'])


# In[11]:

john_word_count_table = john[['word_count']].stack('word_count', new_column_name = ['word','count'])


# In[12]:

john_word_count_table.sort('count',ascending=False)


# In[13]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])


# In[14]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])


# In[15]:

people['tfidf'] = graphlab.text_analytics.tf_idf(people['word_count'])


# In[17]:

john = people[people['name'] == 'Elton John']


# In[18]:

john


# In[19]:

john_tf_idf = john[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# In[20]:

john_tf_idf


# In[21]:

beckham = people[people['name'] == 'Victoria Beckham']


# In[22]:

MaCartney = people[people['name'] == 'Paul McCartney']


# In[23]:

graphlab.distances.cosine(john['tfidf'][0],beckham['tfidf'][0])


# In[25]:

graphlab.distances.cosine(john['tfidf'][0],MaCartney['tfidf'][0])


# In[30]:

knn_model_raw_word_count = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')


# In[33]:

knn_model_raw_word_count.query(john)


# In[34]:

knn_model_tf_idf = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance='cosine')


# In[ ]:




# In[35]:

knn_model_tf_idf.query(john)


# In[36]:

knn_model_raw_word_count.query(beckham)


# In[37]:

knn_model_tf_idf.query(beckham)

