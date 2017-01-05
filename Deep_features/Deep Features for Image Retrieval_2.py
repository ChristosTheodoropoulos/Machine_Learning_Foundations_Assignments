
# coding: utf-8

# #Building an image retrieval system with deep features
# 
# 
# #Fire up GraphLab Create

# In[1]:

import graphlab


# #Load the CIFAR-10 dataset
# 
# We will use a popular benchmark dataset in computer vision called CIFAR-10.  
# 
# (We've reduced the data to just 4 categories = {'cat','bird','automobile','dog'}.)
# 
# This dataset is already split into a training set and test set. In this simple retrieval example, there is no notion of "testing", so we will only use the training data.

# In[2]:

image_train = graphlab.SFrame('image_train_data/')


# #Computing deep features for our images
# 
# The two lines below allow us to compute deep features.  This computation takes a little while, so we have already computed them and saved the results as a column in the data you loaded. 
# 
# (Note that if you would like to compute such deep features and have a GPU on your machine, you should use the GPU enabled GraphLab Create, which will be significantly faster for this task.)

# In[3]:

#deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
#image_train['deep_features'] = deep_learning_model.extract_features(image_train)


# In[4]:

image_train.head()


# #Train a nearest-neighbors model for retrieving images using deep features
# 
# We will now build a simple image retrieval system that finds the nearest neighbors for any image.

# In[5]:

knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],
                                             label='id')


# In[ ]:




# #Use image retrieval model with deep features to find similar images
# 
# Let's find similar images to this cat picture.

# In[6]:

graphlab.canvas.set_target('ipynb')
cat = image_train[18:19]
cat['image'].show()


# In[7]:

knn_model.query(cat)


# We are going to create a simple function to view the nearest neighbors to save typing:

# In[8]:

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


# In[9]:

cat_neighbors = get_images_from_ids(knn_model.query(cat))


# In[10]:

cat_neighbors['image'].show()


# Very cool results showing similar cats.
# 
# ##Finding similar images to a car

# In[11]:

car = image_train[8:9]
car['image'].show()


# In[12]:

get_images_from_ids(knn_model.query(car))['image'].show()


# #Just for fun, let's create a lambda to find and show nearest neighbor images

# In[13]:

show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()


# In[14]:

show_neighbors(8)


# In[15]:

show_neighbors(26)


# In[16]:

my_sarray = graphlab.SArray(image_train['label'])


# In[17]:

sketch = my_sarray.sketch_summary()


# In[18]:

items = ['dog', 'cat', 'automobile', 'bird']


# In[19]:

foo = image_train.filter_by(items, 'label')


# In[20]:

dog = foo[foo['label'] == 'dog']


# In[21]:

cat = foo[foo['label'] == 'cat']


# In[22]:

automobile = foo[foo['label'] == 'automobile']


# In[23]:

bird = foo[foo['label'] == 'bird']


# In[26]:

dog_model = graphlab.nearest_neighbors.create(dog,features=['deep_features'],
                                             label='id')


# In[27]:

cat_model = graphlab.nearest_neighbors.create(cat,features=['deep_features'],
                                             label='id')


# In[28]:

automobile_model = graphlab.nearest_neighbors.create(automobile,features=['deep_features'],
                                             label='id')


# In[29]:

bird_model = graphlab.nearest_neighbors.create(bird,features=['deep_features'],
                                             label='id')


# In[34]:

image_test = graphlab.SFrame('image_test_data/')


# In[38]:

image_test[0:1]


# In[57]:

cat_model.query(image_test[0:1])


# In[58]:

def get_images_from_ids_2(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


# In[59]:

cat_neighbors_ = get_images_from_ids_2(cat_model.query(image_test[0:1]))


# In[60]:

cat_neighbors_['image'].show()


# In[54]:

dog_model.query(image_test[0:1])


# In[55]:

dog_neighbors_ = get_images_from_ids_2(dog_model.query(image_test[0:1]))


# In[56]:

dog_neighbors_['image'].show()


# In[62]:

bar = image_test.filter_by(items, 'label')


# In[63]:

image_test_dog = bar[bar['label'] == 'dog']


# In[64]:

image_test_cat = bar[bar['label'] == 'cat']


# In[65]:

image_test_bird = bar[bar['label'] == 'bird']


# In[66]:

image_test_automobile = bar[bar['label'] == 'automobile']


# In[ ]:



