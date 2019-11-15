#!/usr/bin/env python
# coding: utf-8

# Recepie #1

# In[2]:


from sklearn import tree


# In[3]:


features = [[140, 1], [130, 1], [150, 1], [170, 1]]


# In[4]:


labels = ["apple", "apple", "orange", "orange"]


# In[5]:


clf = tree.DecisionTreeClassifier()


# In[6]:


clf = clf.fit(features, labels)


# In[7]:


print(clf.predict([[146, 0]]))


# Recepie #2

# In[12]:


from sklearn.datasets import load_iris


# In[13]:


iris = load_iris()


# In[36]:


print(iris.data[0])
print(iris.feature_names)
print(iris.target)
print(iris.target_names)


# In[33]:


for i in range(len(iris.target)):
    print ("Example: ", i, "Label: ", iris.target[i], "Features: ", iris.data[i])


# In[ ]:




