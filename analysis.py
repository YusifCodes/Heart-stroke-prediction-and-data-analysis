import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Read csv with pd.read_csv()
df = pd.read_csv("stroke.csv")

# Since the dataframe is not big I will just create a copy to not mess with original data
analysis_df = df.copy()
# Drop rows that have NaN values
analysis_df = analysis_df.dropna()


# Firstly our dataframe is not in such a great shape for data analysis so firstly I want to change that
# We have some binary encoded Yes or No values
# Here I create a simple dictionary to replace those binary encoding with string values
positive_negative_dict = {0:"No", 1:"Yes"}

# Here I actually replace the binary column value data with string
analysis_df["hypertension"] = analysis_df["hypertension"].apply(lambda x: positive_negative_dict[x])
analysis_df["heart_disease"] = analysis_df["heart_disease"].apply(lambda x: positive_negative_dict[x])

# After we are done we will mainly be comparing cases of strokes for men and women
# Here I create two variables stroke_male and stroke_female
# I use DataFrame.iloc[] to search the dataframe where patients suffered a stroke 'analysis_df["stroke"] == 1' with a corresponding gender 'analysis_df["gender"] == "Male/Female"'
stroke_male = analysis_df.loc[(analysis_df["stroke"] == 1) & (analysis_df["gender"] == "Male")] 
stroke_female = analysis_df.loc[(analysis_df["stroke"] == 1) & (analysis_df["gender"] == "Female")] 

# This is a helper function to get the column with a corresponding column_value
# It takes in a column name, the column value, the equals boolean and the dataframe
def get_column_name(column_name, column_value, equals, df):
    # Check if equals is a boolean just for safety, by using python's type()
    if type(equals) == type(True):
        # Simple if statement
        if equals == True:
            return df.loc[df[column_name] == column_value, column_name]
        if equals == False:
            return df.loc[df[column_name] != column_value, column_name]


# Here are the different plots
# I used matplotlib python package


# There are mainly two types of plots here
# A bar chart and a histogram
# The core stays the same just the data changes

# 1 Age comparison histogram
# plt.hist() creates a histogram 
# We pass our input data and the two corresponding labels
plt.hist([stroke_male["age"], stroke_female["age"]], label=["Male", "Female"])
# Label for x axis
plt.xlabel("Age")
# Label for y axis
plt.ylabel("Number of cases")
# Plot title
plt.suptitle("Stroke age comparison between men and women")
# This is used to place a legend on the Axes
plt.legend()
# Show the histogram
plt.show()


# 2 Smoking comparison histogram
plt.hist([stroke_male.loc[stroke_male["smoking_status"] != "Unknown", "smoking_status"], stroke_female.loc[stroke_female["smoking_status"] != "Unknown", "smoking_status"]], label=["Male", "Female"])
plt.ylabel("Number of cases")
plt.suptitle("Stroke and smoking relationship comparison between men and women")
plt.legend()
plt.show()

# 3 Hypertension bar chart
# We need to define the width of the bar
width = 0.3       
# We need to define the amount of bars we need for our case since it is yes or no we only need two 
# This is an x value
ind = np.arange(2)
# We create a bar, pass it the x , then the y - bar data
plt.bar(ind, [get_column_name("hypertension", "Yes", True, stroke_male).shape[0],get_column_name("hypertension", "No", True, stroke_male).shape[0]], width, label='Male')
# Here you can notice we add width to ind, it is done so the bars do not overlap
plt.bar(ind + width, [get_column_name("hypertension", "Yes", True, stroke_female).shape[0], get_column_name("hypertension", "No", True, stroke_female).shape[0]], width, label='Female')
# Position the xlabels accordingly
plt.xticks(ind + width / 2, ('Hypertension', 'No hypertension'))

plt.suptitle("Stroke and hypertension relationship comparison between men and women")
# This is used to place a legend on the Axes
# loc='best' automatically chooses the best position
plt.legend(loc='best')
# Show the data
plt.show()


# 4 Heart disease
width = 0.3       
ind = np.arange(2)
plt.bar(ind, [get_column_name("heart_disease", "Yes", True, stroke_male).shape[0],get_column_name("heart_disease", "No", True, stroke_male).shape[0]], width, label='Male')
plt.bar(ind + width, [get_column_name("heart_disease", "Yes", True, stroke_female).shape[0], get_column_name("heart_disease", "No", True, stroke_female).shape[0]], width, label='Female')
plt.xticks(ind + width / 2, ('Suffered from a heart disease', 'Did not suffer from a heart disease'))
plt.suptitle("Stroke and heart disease relationship comparison between men and women")
plt.xlabel("Cases")
plt.legend(loc='best')
plt.show()


# 5 Marriage
width = 0.3       
ind = np.arange(2)
plt.bar(ind, [get_column_name("ever_married", "Yes", True, stroke_male).shape[0],get_column_name("ever_married", "No", True, stroke_male).shape[0]], width, label='Male')
plt.bar(ind + width, [get_column_name("ever_married", "Yes", True, stroke_female).shape[0], get_column_name("ever_married", "No", True, stroke_female).shape[0]], width, label='Female')
plt.xticks(ind + width / 2, ('Was/Is married', 'Never married'))
plt.suptitle("Stroke and marriage relationship comparison between men and women")
plt.xlabel("Cases")
plt.legend(loc='best')
plt.show()

# 6 Work type
plt.hist([stroke_male["work_type"], stroke_female["work_type"]], label=["Male", "Female"])
plt.suptitle("Stroke and work type relationship comparison between men and women")
plt.legend()
plt.show()

# 7 Residence type
width = 0.3  
ind = np.arange(2)
plt.bar(ind, [get_column_name("Residence_type", "Urban", True, stroke_male).shape[0],get_column_name("Residence_type", "Rural", True, stroke_male).shape[0]], width, label='Male')
plt.bar(ind + width, [get_column_name("Residence_type", "Urban", True, stroke_female).shape[0], get_column_name("Residence_type", "Rural", True, stroke_female).shape[0]], width, label='Female')
plt.xticks(ind + width / 2, ('Urban', 'Rural'))
plt.suptitle("Stroke and residence type relationship comparison \n between men and women")
plt.xlabel("Cases")
plt.legend(loc='best')
plt.show()

# 8 Bmi
plt.hist([stroke_male["bmi"], stroke_female["bmi"]], label=["Male", "Female"])
plt.suptitle("Stroke and bmi relationship comparison between men and women")
plt.xlabel("BMI")
plt.ylabel("Cases")
plt.legend()
plt.show()

# 9 Avg glucose level 
plt.hist([stroke_male["avg_glucose_level"], stroke_female["avg_glucose_level"]], label=["Male", "Female"])
plt.suptitle("Stroke and average glucose level relationship comparison \n between men and women")
plt.xlabel("Average glucose level (mmol/L)")
plt.ylabel("Cases")
plt.legend()
plt.show()