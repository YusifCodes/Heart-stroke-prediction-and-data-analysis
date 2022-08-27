import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

#  Read csv with pd.read_csv()
df = pd.read_csv("stroke.csv")

# Since the dataframe is not big I will just create a copy to not mess with original data
norm_df = df.copy()
# Drop rows that have NaN values
norm_df = norm_df.dropna()

# Here I start making the data understandable for our keras model

# Here I define a method that takes in a list of column values (A list of strings) and return a dictionary where each column value got assigned a number
def create_int_dict(query_list):
    dict = {}
    i = 0
    for query in query_list:
        dict[query] = i
        i+=1
    return dict

# Here I call the method defined earlier for every unique column
# i used the in-built pandas .unique() method which automatically returns, you guessed it, all unique values in a column
smoking_dict = create_int_dict(norm_df["smoking_status"].unique())
residence_dict = create_int_dict(norm_df["Residence_type"].unique())
work_dict = create_int_dict(norm_df["work_type"].unique()) 
gender_dict = create_int_dict(norm_df["gender"].unique())
marriage_dict = create_int_dict(norm_df["ever_married"].unique())

# Another helper method to actually replace the column value with the numbers they got assigned earlier with create_int_dict()
# This method takes in the column, the corresponding dictionary, and the dataframe
def replace_column_value(column, dict, df):
    df[column] = df[column].apply(lambda x: dict[x])

# Here I call the function defined earlier
replace_column_value("gender", gender_dict, norm_df)
replace_column_value("work_type", work_dict, norm_df)
replace_column_value("Residence_type", residence_dict, norm_df)
replace_column_value("smoking_status", smoking_dict, norm_df)
replace_column_value("ever_married", marriage_dict, norm_df)

# This is an application of train_test_split() from sklearn
# What it does is it splits arrays or matrices into random train and test subsets

# Overall we need 3 dataframes - Training, testing and validation dataframes

# Firstly we create train_df and test_df, we give the dataframe we want to split, which is norm_df, after that we have test_size=0.2 which means that the method will shuffle and choose 20% percent of the data
train_df, test_df = train_test_split(norm_df, test_size=0.2)
# Secondly we overwrite train_df and create a new variable val_df, then we pass the old train_df instead of norm_df, and test_size stays 20%
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Then we need x and y values
# X is input, y is output

# After that we just use .pop() method and pass it the column value which can be either 0 or 1 (1 - patient suffered a stroke, 0 - patient did not suffer a stroke)
# .pop() returns the item and removes it from the frame
# np.array() return a numpy array
train_y = np.array(train_df.pop('stroke'))
val_y = np.array(val_df.pop('stroke'))
test_y = np.array(test_df.pop('stroke'))

# Since we popped the stroke column we don't have to worry about it and just pass in the dataframes
train_x = np.array(train_df)
val_x = np.array(val_df)
test_x = np.array(test_df)

# Just a method to retur a model created in it
def create_model():

    # We use a normal keras sequential model
    # It uses a sequential model API which allows to instatiate a model and the add a bunch of linear layers to it
    model = keras.Sequential([

        # In a model we can split it into 3 parts
        # Input layer, hidden layers, output layer

        # First layer is a dense input layer
        # Dense layers are the most widely used kind of layer
        # It represents a layer of neurons which receive input from from all neurons of the previous layer
        # We need to give it some parameters
        # The first paramater is units, it defines the size of the output from the dense layer
        # Second is input shape, we just use .shape which return a list which looks like this [rows, columns] that's why I use [-1] index to get the last elemnt oof the list
        # Third is the activation function, I used relu for this case
        # Activation function is a mathematical “gate” in between the input feeding the current neuron and its output going to the next layer
        keras.layers.Dense(64, input_shape=[train_x.shape[-1]], activation="relu"),
        # Now that we defined the input layer, we can move on to our hidden layers
        # Since we don't have that much data we do not need that many layers, one will do the job
        # It has the sama params as the previous layer with the exception of input_shape (We need to define it only once in the input layer)
        keras.layers.Dense(64, activation="relu"),
        # Here we have our output layer, which is a little different
        # Firstly our units are not 64 but 1 this time, simply because the output shape for our case scenario is [1,] since it can only be 1 or 0
        # If you had more output cases for example5 your units would be 5, since the output would have the shape [5,]
        # And you should use a different activation func for outpur layers too
        # I used sigmoid since its good for binary classification
        keras.layers.Dense(1, activation="sigmoid")
    ])

    # After we defined the model and the layers we can compile our model
    model.compile(
        # You have to pass some main parameters
        # First one is a loss function, I used binary_crossentropy since we are doing linear binary classification
        # A loss func is basicallly a method of evaluating how well your algorithm models your dataframe
        loss="binary_crossentropy",
        # Then we have an optimizer, safe bet is to use adam
        # Optimizers are algorithms or methods used to minimize a loss function or to maximize the efficiency of production
        optimizer="adam",
        # And the we just pass in the metrics that interest us
        # I just used BinaryAccuracy
        # It shows how accurate is the model with its decision making
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy')
            ]
        )

    # We return our compiled model
    return model

# Call the function and store the model in a variable
model = create_model()
# Here I define an early stopping callback
# It is used to stop model training if it stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    # This is the metric that interests us
    # If validation accuracy stops improving we are done with training
    monitor='val_accuracy', 
    # This is logging
    verbose=1,
    # How many epochs can pass without improvement before we stop
    patience=10,
    # "max" mode training will stop when the quantity monitored has stopped increasing
    mode='max',
    # Whether to restore model weights from the epoch with the best value of the monitored quantity
    restore_best_weights=True)

# Then we fit the model 
# We pass x, y, epochs is how many times the model will train, batch_size is the size of data batches that will be given to our model, then pass in the early stopping callback and validation data
# You can tune your model based on results of the metrics of validation data (That's what "val_accuracy" in the callback meant, we stop when the validation data's accuracy metric stops improving)
model.fit(train_x, train_y, epochs=50, batch_size=24, callbacks=[early_stopping], validation_data=[val_x, val_y])

# After we train the model we can evaluate how good our model did with model.evaluate
# We give it fresh test data we prepared earlier 
# And it runs through our model and gives us a list of metrics, so basically a score
score = model.evaluate(test_x, test_y)

# This is just for logging the final score
# I use python's zip() which creates tuples 
for name,val in zip(model.metrics_names, score):
    # Log it
    print(f"{name}: {val}")

# Voila - 95% accuracy model