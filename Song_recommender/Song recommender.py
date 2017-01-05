
# coding: utf-8

# #Building a song recommender
# 
# 
# #Fire up GraphLab Create

# In[19]:

import graphlab


# #Load music data

# In[20]:

song_data = graphlab.SFrame('song_data.gl/')


# #Explore data
# 
# Music data shows how many times a user listened to a song, as well as the details of the song.

# In[21]:

song_data.head()


# ##Showing the most popular songs in the dataset

# In[22]:

graphlab.canvas.set_target('ipynb')


# In[23]:

song_data['song'].show()


# In[24]:

len(song_data)


# ##Count number of unique users in the dataset

# In[25]:

users = song_data['user_id'].unique()


# In[26]:

len(users)


# #Create a song recommender

# In[27]:

train_data,test_data = song_data.random_split(.8,seed=0)


# ##Simple popularity-based recommender

# In[28]:

popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id='user_id',
                                                         item_id='song')


# ###Use the popularity model to make some predictions
# 
# A popularity model makes the same prediction for all users, so provides no personalization.

# In[29]:

popularity_model.recommend(users=[users[0]])


# In[30]:

popularity_model.recommend(users=[users[1]])


# ##Build a song recommender with personalization
# 
# We now create a model that allows us to make personalized recommendations to each user. 

# In[31]:

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')


# ###Applying the personalized model to make song recommendations
# 
# As you can see, different users get different recommendations now.

# In[32]:

personalized_model.recommend(users=[users[0]])


# In[33]:

personalized_model.recommend(users=[users[1]])


# ###We can also apply the model to find similar songs to any song in the dataset

# In[34]:

personalized_model.get_similar_items(['With Or Without You - U2'])


# In[35]:

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])


# #Quantitative comparison between the models
# 
# We now formally compare the popularity and the personalized models using precision-recall curves. 

# In[36]:

if graphlab.version[:3] >= "1.6":
    model_performance = graphlab.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
    graphlab.show_comparison(model_performance,[popularity_model, personalized_model])
else:
    get_ipython().magic(u'matplotlib inline')
    model_performance = graphlab.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample=.05)


# The curve shows that the personalized model provides much better performance. 

# In[ ]:

# Solving Quiz


# In[40]:

kanye_west_listeners = song_data[song_data['artist'] == 'Kanye West'] 


# In[42]:

kanye_west_listeners_unigue = kanye_west_listeners['user_id'].unique()


# In[43]:

len(kanye_west_listeners_unigue)


# In[45]:

foo_fighters_listeners = song_data[song_data['artist'] == 'Foo Fighters'] 


# In[46]:

foo_fighters_listeners_unigue = foo_fighters_listeners['user_id'].unique()


# In[47]:

len(foo_fighters_listeners_unigue)


# In[49]:

taylor_swift_listeners = song_data[song_data['artist'] == 'Taylor Swift'] 


# In[50]:

taylor_swift_listeners_unigue = taylor_swift_listeners['user_id'].unique()


# In[51]:

len(taylor_swift_listeners_unigue)


# In[52]:

lady_gaga_listeners = song_data[song_data['artist'] == 'Lady GaGa'] 


# In[53]:

lady_gaga_listeners_unigue = lady_gaga_listeners['user_id'].unique()


# In[54]:

len(lady_gaga_listeners_unigue)


# In[56]:

artists = song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})


# In[57]:

artists


# In[60]:

artists = artists.sort('total_count')


# In[61]:

artists


# In[66]:

artists.tail()


# In[67]:

item_similarity_recommender = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')


# In[68]:

subset_test_users = test_data['user_id'].unique()[0:10000]


# In[70]:

item_similarity_recommender.recommend(subset_test_users,k=1)


# In[71]:

operations={'total_count': graphlab.aggregate.SUM('listen_count')}


# In[72]:

operations={'count': graphlab.aggregate.COUNT()}


# In[73]:

key_columns='song'


# In[77]:



