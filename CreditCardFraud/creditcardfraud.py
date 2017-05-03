import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')

#Select only the anonymized features.
v_features = df.ix[:,1:29].columns

# create distribution graph for every feature
for i, cn in enumerate(df[v_features]):
    ax = plt.subplot()
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()

# remove all the features with similar graphs
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)

# create new fearures for distribution
df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0

#Rename 'Class' to 'Fraud'.
df = df.rename(columns={'Class': 'Fraud'})

#create Fraud and normal feature distribution
Fraud = df[df.Fraud == 1]
Normal = df[df.Normal == 1]

# create X_train by taking 80% of fraud transactions and 80% of normal transactions
X_train = Fraud.sample(frac=0.8)
count_Frauds = len(X_train)
X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)
X_test = df.loc[~df.index.isin(X_train.index)]

# create Y_train by taking 80% of fraud transactions and 80% of normal transactions
y_train = X_train.Fraud
y_train = pd.concat([y_train, X_train.Normal], axis=1)
y_test = X_test.Fraud
y_test = pd.concat([y_test, X_test.Normal], axis=1)

# drop the guest features
X_train = X_train.drop(['Fraud','Normal'], axis = 1)
X_test = X_test.drop(['Fraud','Normal'], axis = 1)

# ratio = len(X_train)/count_Frauds
# y_train.Fraud *= ratio
# y_test.Fraud *= ratio

# scale values for features
features = X_train.columns.values
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std

#split the dataset for train,test & validation
split = int(len(y_test)/2)

inputX = X_train.as_matrix()
print inputX.shape
inputY = y_train.as_matrix()
inputX_valid = X_test.as_matrix()[:split]
inputY_valid = y_test.as_matrix()[:split]
inputX_test = X_test.as_matrix()[split:]
inputY_test = y_test.as_matrix()[split:]

#parameters
learning_rate = 0.005
training_epoch = 10
batch_size = 2048
display_step = 1

#tf graph input
x = tf.placeholder(tf.float32,[None,19])
y = tf.placeholder(tf.float32,[None,2])

#set model weights
w = tf.Variable(tf.zeros([19,2]))
b = tf.Variable(tf.zeros([2]))

#construct model
pred = tf.nn.softmax(tf.matmul(x,w) + b) #softmax activation

#minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))

#Gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#initializing variables
init = tf.global_variables_initializer()

print inputX[1000:1001]

#launch the graph
with tf.Session() as sess:
    sess.run(init)
    final_output_array = []
    #training cycle
    for epoch in range(training_epoch):
        total_batch = len(inputX)/batch_size
        avg_cost = 0
        #loop over all the batches
        for batch in range(total_batch):
            batch_xs = inputX[(batch)*batch_size:(batch+1) *batch_size]
            batch_ys = inputY[(batch)*batch_size:(batch+1) *batch_size]

            # run optimizer and cost operation
            _,c= sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost += c/total_batch

        correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #disply log per epoch step
        if (epoch+1) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX_test,y: inputY_test})
            print "epoch:",epoch+1,"train_accuracy:",train_accuracy,"cost:",newCost,"valid_accuracy:",sess.run([accuracy],feed_dict={x:inputX_valid,y:inputY_valid})
            output = sess.run(pred,feed_dict={x:inputX[1000:1001]})
            final_output_array.append(output)
            print

    final_output = sum(final_output_array)/epoch
    # print final_output
    # print
    # print final_output[:,:1]
    # print final_output[:,:2]
    output = 'fraud' if final_output[:,:1] > final_output[:,1:2] else 'normal'
    print output
    print 'optimization finished.'
