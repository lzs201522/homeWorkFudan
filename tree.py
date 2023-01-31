import pandas as pd
from sklearn.model_selection import train_test_split
# This is used to split our data into training and testing sets
from sklearn import tree # Here tree is a module
from sklearn.metrics import accuracy_score
# Used to check the goodness of our model
import matplotlib.pyplot as plt
# Used to plot figures

df1=pd.read_csv("test.csv")
# storing our excel file in df1
df1.info() # This function is used to check whether our data consists of any missing or null values
X=df1.loc[:,df1.columns!="Buying" ]
X=X.loc[:,X.columns!="User id"]
y=df1["Buying"]




X_train, X_test, Y_train, Y_test=train_test_split(X, y, test_size=0.2, random_state=0)
# Here test_size = 0.2 means it uses 20% of our input data for testing and 80% for training set
# random_state = 0 means every time it uses the same set of testing and training set for evaluation

clftree1=tree.DecisionTreeClassifier(criterion="entropy")
# Using Entropy for computing the Decision Tree
clftree1.fit(X,y)
pred=clftree1.predict(X_test)    # Predicting the values for our test data
accuracy_score1=accuracy_score(Y_test, pred)   # Finding the accuracy score of our model
print(accuracy_score1)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10),dpi=300)
# Let us create a figure with size (10X10) and density per inch = 300
tree.plot_tree(clftree1, feature_names=list(df1.columns),class_names="01",filled =True)
# plot_tree is used to plot our decision tree. The parameters are our Decision Tree, feature names, class names to be displayed in
  # string format (or) as a list, filled=True will automatically fill colours to our tree etc
fig.savefig("imagename1.jpeg.png")                                     

clftree2=tree.DecisionTreeClassifier(criterion="gini")
# Using Gini Index for computing the Decision Tree
clftree2.fit(X,y)
pred=clftree2.predict([[50,1,0,1]])    # Predicting the values for our test data
accuracy_score2=accuracy_score(Y_test, pred)   # Finding the accuracy score of our model
print(accuracy_score2)

fig, ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10),
dpi=300)
tree.plot_tree(clftree2, feature_names=list(df1.columns),
class_names="01", filled=True)
fig.savefig('imagename2.jpeg.png')

