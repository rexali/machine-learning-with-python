# Load the necesssary Library
from sklearn.model_selection import train_test_split ,StratifiedShuffleSplit
from sklearn.datasets import load_iris
#from sklearn import StratifiedShuffleSplit
import numpy as np


# Load iris dataset
data = load_iris()

# Extract the dependend and independent variables
# y is our class label
# x is our instances/records
x    = data['data']
y    = data['target']

def get_iris_data():
    """
    Returns Iris dataset
    """
    # Load iris dataset
    data = load_iris()

    # Extract the dependend and independent variables
    # y is our class label
    # x is our instances/records
    x    = data['data']
    y    = data['target']
    # For ease we merge them
    # column merge
    input_dataset = np.column_stack([x,y])

    # Let us shuffle the dataset
    # We want records distributed randomly
    # between our test and train set
    np.random.shuffle(input_dataset)

    return input_dataset

# We need  80/20 split.
# 80% of our records for Training
# 20% Remaining for our Test set
train_size = 0.8
test_size  = 1-train_size

# get the data
input_dataset = get_iris_data()
# Split the data
train,test = train_test_split(input_dataset,test_size=test_size)
# Print the size of original dataset
print ("Dataset size ", input_dataset.shape)
# Print the train/test split
print ("Train size ",train.shape)
print ("Test  size ",test.shape)


#  Let's see if the class labels are proportionately distributed between 
# the training and the test sets. 
# This is a typical class imbalance problem:

def get_class_distribution(y):
    """
Given an array of class labels
Return the class distribution
"""
    distribution = {}
    set_y = set(y)
    for y_label in set_y:
        no_elements = len(np.where(y == y_label)[0])
        distribution[y_label] = no_elements
    dist_percentage = {class_label: count/(1.0*sum(distribution.values())) for class_label,count in distribution.items()}
    return dist_percentage

def print_class_label_split(train,test):
    """
  Print the class distribution
  in test and train dataset
  """
    y_train = train[:,-1]
    train_distribution = get_class_distribution(y_train)
    print ("\nTrain data set class label distribution")
    print ("=========================================\n")

    for k,v in train_distribution.items():
        print ("Class label =%d, percentage records =%.2f"%(k,v))

    y_test = test[:,-1]    
    test_distribution = get_class_distribution(y_test)
    
    print ("\nTest data set class label distribution")
    print ("=========================================\n")
    
    for k,v in test_distribution.items():
        print ("Class label =%d, percentage records =%.2f"%(k,v))

print_class_label_split(train,test)

# Let's see how we distribute the class labels uniformly 
# between the train and the test sets:

# Perform Split the data
#stratified_split = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=0)
#no_of_split = stratified_split.get_n_splits(input_dataset)
#for train_indx,test_indx in stratified_split:
    #train = input_dataset[train_indx]
    #test =  input_dataset[test_indx]
    #print_class_label_split(train,test)