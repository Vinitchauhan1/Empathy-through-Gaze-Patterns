# %% [markdown]
# # Data Analysis and Modeling Setup
# 
# In this section, we've set up an environment for data analysis, visualization, and modeling using Python libraries.
# 
# ## Import Libraries

# %%
#for the exploratory data analysis purpose and also for vizualization
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#for modelling and finding out the correlation in the dataset
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#libraries which are used in the modellling purpose.
import tensorflow as tf

# %% [markdown]
# ## Load and Combine Raw Data 
# # Directory containing the TSV files with raw eye-tracking data

# %%
# Directory containing the TSV files with raw eye-tracking data
directory = "E:\\ce888\\raw_data"

# Create an empty list to store the loaded dataframes
df_list = []

# Get a list of all TSV files in the directory
file_list = glob.glob(directory + '\\*.tsv')

# Load all files into a list of DataFrames
for filepath in file_list:
    df = pd.read_csv(filepath, delimiter='\t', low_memory=False)
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
df_all = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame to a CSV file
df_all.to_csv('combined_file.csv', index=False)


# %% [markdown]
# #Load dataset and display basic information

# %%
data = pd.read_csv('combined_file.csv')

# %% [markdown]
# Data Information

# %%
data.info()

# %%
data.head()

# %%
data.tail()

# %%
#view summary statistics of the dataset
data.describe()

# %%
#define the shape of the data
data.shape 

# %%
data.columns

# %% [markdown]
# Data Transformation
# 

# %%
data['Pupil diameter right'] = data['Pupil diameter left'].str.replace(',', '.').astype(float)
data['Pupil diameter left'] = data['Pupil diameter left'].str.replace(',', '.').astype(float)
data['Eye position left X (DACSmm)'] = data['Eye position left X (DACSmm)'].str.replace(',', '.').astype(float)
data['Eye position left Y (DACSmm)'] = data['Eye position left Y (DACSmm)'].str.replace(',', '.').astype(float)
data['Eye position left Z (DACSmm)'] = data['Eye position left Z (DACSmm)'].str.replace(',', '.').astype(float)
data['Eye position right X (DACSmm)'] = data['Eye position right X (DACSmm)'].str.replace(',', '.').astype(float)
data['Eye position right Y (DACSmm)'] = data['Eye position right Y (DACSmm)'].str.replace(',', '.').astype(float)
data['Eye position right Z (DACSmm)'] = data['Eye position right Z (DACSmm)'].str.replace(',', '.').astype(float)
data['Gaze point left X (DACSmm)'] = data['Gaze point left X (DACSmm)'].str.replace(',', '.').astype(float)
data['Gaze point left Y (DACSmm)'] = data['Gaze point left Y (DACSmm)'].str.replace(',', '.').astype(float)
data['Gaze point right X (DACSmm)'] = data['Gaze point right X (DACSmm)'].str.replace(',', '.').astype(float)
data['Gaze point right Y (DACSmm)'] = data['Gaze point right Y (DACSmm)'].str.replace(',', '.').astype(float)
data['Gaze point X (MCSnorm)'] = data['Gaze point X (MCSnorm)'].str.replace(',', '.').astype(float)
data['Gaze point Y (MCSnorm)'] = data['Gaze point Y (MCSnorm)'].str.replace(',', '.').astype(float)
data['Gaze point left X (MCSnorm)'] = data['Gaze point left X (MCSnorm)'].str.replace(',', '.').astype(float)
data['Gaze point left Y (MCSnorm)'] = data['Gaze point left Y (MCSnorm)'].str.replace(',', '.').astype(float)
data['Gaze point right X (MCSnorm)'] = data['Gaze point right X (MCSnorm)'].str.replace(',', '.').astype(float)
data['Gaze point right Y (MCSnorm)'] = data['Gaze point right Y (MCSnorm)'].str.replace(',', '.').astype(float)
data['Fixation point X (MCSnorm)'] = data['Fixation point X (MCSnorm)'].str.replace(',', '.').astype(float)
data['Fixation point Y (MCSnorm)'] = data['Fixation point Y (MCSnorm)'].str.replace(',', '.').astype(float)
data['Gaze direction left X'] = data['Gaze direction left X'].str.replace(',', '.').astype(float)
data['Gaze direction left Y'] = data['Gaze direction left Y'].str.replace(',', '.').astype(float)
data['Gaze direction left Z'] = data['Gaze direction left Z'].str.replace(',', '.').astype(float)
data['Gaze direction right X'] = data['Gaze direction right X'].str.replace(',', '.').astype(float)
data['Gaze direction right Y'] = data['Gaze direction right Y'].str.replace(',', '.').astype(float)
data['Gaze direction right Z'] = data['Gaze direction right Z'].str.replace(',', '.').astype(float)

# %%
# Convert columns to datetime format with day-first indication
data['Recording date'] = pd.to_datetime(data['Recording date'], dayfirst=True)
data['Export date'] = pd.to_datetime(data['Export date'], dayfirst=True)

# %%
format_string = "%Y-%m-%d"
data['Export date'] = pd.to_datetime(data['Export date'], errors='coerce')
data['Recording date'] = pd.to_datetime(data['Export date'], errors='coerce')
data['Recording date UTC'] = pd.to_datetime(data['Recording date UTC'], errors='coerce')
data['Recording start time'] = pd.to_datetime(data['Recording start time'], errors='coerce')
data['Recording start time UTC'] = pd.to_datetime(data['Recording start time UTC'], errors='coerce')

# %%
participant_dict = {}
for i in range(1, 61):
    participant_dict[f"Participant{i:04d}"] = i

data["Participant name"] = data["Participant name"].map(participant_dict)

# %%
data_csv_score = pd.read_csv("E:\\ce888\\questionnaires\\Questionnaire_datasetIB.csv", encoding='latin1')
data_csv_score = data_csv_score.rename(columns={"Participant nr": "Participant name"})

# %%
# Merge the two datasets based on the Participant name column
data = pd.merge(data, data_csv_score[['Participant name', 'Total Score extended']], on='Participant name', how='left')

# Set the timestamp column as the index of the DataFrame
data.set_index("Recording timestamp", inplace=True)

# %%
data

# %%
data.to_csv("tsv_data_with_score.csv", index=False)

print("CSV file 'tsv_data_with_score.csv' saved successfully.")


# %% [markdown]
# # Working on EYETzip

# %%
# Get a list of all the .csv files in the folder 
filenames = glob.glob("E:\ce888\EYET\*.csv")

Experiment_count = len(filenames) 
print ("eye-gaze trajectories : ", Experiment_count)

# %%
#read and merged csv file
path = "E:\ce888\EYET" 
all_files = os.listdir(path)

df_list = [] 
for filename in all_files:
    if filename.endswith(".csv"):
        file_path = os.path.join(path, filename)
        df = pd.read_csv(file_path)
        df_list.append(df)
df = pd.concat(df_list)

# %% [markdown]
# #Saving and Loading Data

# %%
# Save the merged DataFrame to a CSV file
df.to_csv("merged_eyetzip_data.csv", index=False)

# %%
# Read the CSV file
main_csv = pd.read_csv("E:\\ce888\\merged_eyetzip_data.csv")

# %%
main_csv

# %% [markdown]
# #Convert string numbers with comma as decimal separator to float.
# As most features  data type is string we need to convert it to float and change , to . due to which it is interpreted as object

# %%
main_csv['Pupil diameter right'] = main_csv['Pupil diameter left'].str.replace(',', '.').astype(float)
main_csv['Pupil diameter left'] = main_csv['Pupil diameter left'].str.replace(',', '.').astype(float)
main_csv['Eye position left X (DACSmm)'] = main_csv['Eye position left X (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Eye position left Y (DACSmm)'] = main_csv['Eye position left Y (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Eye position left Z (DACSmm)'] = main_csv['Eye position left Z (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Eye position right X (DACSmm)'] = main_csv['Eye position right X (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Eye position right Y (DACSmm)'] = main_csv['Eye position right Y (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Eye position right Z (DACSmm)'] = main_csv['Eye position right Z (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point left X (DACSmm)'] = main_csv['Gaze point left X (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point left Y (DACSmm)'] = main_csv['Gaze point left Y (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point right X (DACSmm)'] = main_csv['Gaze point right X (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point right Y (DACSmm)'] = main_csv['Gaze point right Y (DACSmm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point X (MCSnorm)'] = main_csv['Gaze point X (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point Y (MCSnorm)'] = main_csv['Gaze point Y (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point left X (MCSnorm)'] = main_csv['Gaze point left X (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point left Y (MCSnorm)'] = main_csv['Gaze point left Y (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point right X (MCSnorm)'] = main_csv['Gaze point right X (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Gaze point right Y (MCSnorm)'] = main_csv['Gaze point right Y (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Fixation point X (MCSnorm)'] = main_csv['Fixation point X (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Fixation point Y (MCSnorm)'] = main_csv['Fixation point Y (MCSnorm)'].str.replace(',', '.').astype(float)
main_csv['Gaze direction left X'] = main_csv['Gaze direction left X'].str.replace(',', '.').astype(float)
main_csv['Gaze direction left Y'] = main_csv['Gaze direction left Y'].str.replace(',', '.').astype(float)
main_csv['Gaze direction left Z'] = main_csv['Gaze direction left Z'].str.replace(',', '.').astype(float)
main_csv['Gaze direction right X'] = main_csv['Gaze direction right X'].str.replace(',', '.').astype(float)
main_csv['Gaze direction right Y'] = main_csv['Gaze direction right Y'].str.replace(',', '.').astype(float)
main_csv['Gaze direction right Z'] = main_csv['Gaze direction right Z'].str.replace(',', '.').astype(float)

# %% [markdown]
# #Date Parsing : Preprocessing to ensure that the date and time variables are in a consistent and machine-readable format

# %%
format_string = "%Y-%m-%d"
main_csv['Export date'] = pd.to_datetime(main_csv['Export date'], errors='coerce')
main_csv['Recording date'] = pd.to_datetime(main_csv['Export date'], errors='coerce')
main_csv['Recording date UTC'] = pd.to_datetime(main_csv['Recording date UTC'], errors='coerce')
main_csv['Recording start time'] = pd.to_datetime(main_csv['Recording start time'], errors='coerce')
main_csv['Recording start time UTC'] = pd.to_datetime(main_csv['Recording start time UTC'], errors='coerce')

# %% [markdown]
# #Coverting participant name from Participant0001 to 1 

# %%
participant_dict = {}
for i in range(1, 61):
    participant_dict[f"Participant{i:04d}"] = i

main_csv["Participant name"] = main_csv["Participant name"].map(participant_dict)

# %% [markdown]
# Load Questinonnaire File

# %%
main_csv_score = pd.read_csv("E:\\ce888\\questionnaires\\Questionnaire_datasetIB.csv", encoding='latin1')
main_csv_score = main_csv_score.rename(columns={"Participant nr": "Participant name"})

# %%
# Merge the two datasets based on the Participant name column
main_csv = pd.merge(main_csv, main_csv_score[['Participant name', 'Total Score extended']], on='Participant name', how='left')

# Set the timestamp column as the index of the DataFrame
main_csv.set_index("Recording timestamp", inplace=True)

# %%
main_csv

# %% [markdown]
# Save the main_csv DataFrame to a new CSV file named "final_data.csv"

# %%
main_csv.to_csv("eyetzip_data_with_score.csv", index=False)

print("CSV file 'eyetzip_data_with_score.csv' saved successfully.")


# %% [markdown]
# #  Data Analysis
# # Column of interest

# %%
def analyze_gaze_duration(data):
    """Analyze the 'Gaze event duration' column."""
    # Check for missing values
    missing_values = data['Gaze event duration'].isnull().sum()
    print(f"Missing values in 'Gaze event duration': {missing_values}")
    
    # Display the first few entries
    print("\nFirst few entries of 'Gaze event duration':")
    print(df['Gaze event duration'].head())
    
    # Calculate the average gaze duration per eye movement type
    avg_gaze_duration = data.groupby('Eye movement type')['Gaze event duration'].mean()
    print("\nAverage 'Gaze event duration' per 'Eye movement type':")
    print(avg_gaze_duration)

analyze_gaze_duration(data)


# %%
data.columns

# %%
import datetime
data['Recording date']=pd.to_datetime(data['Recording date'])
data['Recording start time']=pd.to_datetime(data['Recording start time'])
data['Total Score extended']=data['Total Score extended'].astype(int)
data

# %% [markdown]
# # Performing visual Exploratory data analysis on Row Data

# %%
sns.countplot(x='Eye movement type', data=data)
plt.title("Distribution of Eye Movement Types")
plt.xlabel("Eye Movement Type")
plt.ylabel("Count of Observations")

# Display plot
plt.show()


# %%
plt.hist(data['Recording duration'], bins=30, edgecolor='black')
plt.xlabel('Recording Duration (Seconds)')
plt.ylabel('Number of Observations')
plt.title('Distribution of Recording Durations')

plt.show()


# %%
# Scatter plot to visualize the distribution of gaze points on the screen
plt.scatter(data['Gaze point X'], data['Gaze point Y'], alpha=0.5)
plt.xlabel('Gaze Point X (Horizontal Position)')
plt.ylabel('Gaze Point Y (Vertical Position)')
plt.title('Distribution of Gaze Points on Screen')

plt.show()

# %%
plt.figure(figsize=(14, 7))
sns.boxplot(x=data['Project name'], y=data['Eye movement type index'])
plt.xlabel('Participant Name')
plt.ylabel('Eye Movement Type Index')
plt.title('Distribution of Eye Movement Type Index by Participant')
plt.xticks(rotation=90)

plt.show()

# %%
pd.DataFrame(data['Total Score extended'].value_counts()).plot(kind="bar", figsize=(14,3))   

# %% [markdown]
# # Correlation Matrix for Row Data

# %%
# Filter only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Compute correlation on numeric columns
corr_matrix = numeric_data.corr()

f, ax = plt.subplots(figsize=(12, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap=cmap)
plt.show()


# %%
# Select relevant columns
selected_columns = ['Project name','Total Score extended', 'Gaze point X', 'Gaze point Y', 'Gaze event duration']


# Subset the DataFrame
subset_df = data[selected_columns]

# Correlation analysis
correlation_matrix = one_hot_encoded.corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
#making a copy of cleaned data that will be use further for the analysis purpose
new_data_a = data.copy()

# %% [markdown]
# #  Differentiate into two groups based on project names

# %%
# Count occurrences of each project name in the 'Project Name' column
project_counts = data['Project name'].value_counts()
# Print the counts for "Control group experiment" and "Test group experiment"
print(f'Control group experiment: {project_counts.get("Control group experiment", 0)}')
print(f'Test group experiment: {project_counts.get("Test group experiment", 0)}')

# %%
# Separate data based on project name
control_group_data = data[data['Project name'] == 'Control group experiment']
test_group_data = data[data['Project name'] == 'Test group experiment']

# Save the separated data to CSV files
control_group_data.to_csv('control_group_data.csv', index=False)
test_group_data.to_csv('test_group_data.csv', index=False)

# Print some summary statistics for each group
print("Control Group Summary:")
print(control_group_data.describe())

print("\nTest Group Summary:")
print(test_group_data.describe())


# %%
# Load the separated data
control_group_data = pd.read_csv('control_group_data.csv')
test_group_data = pd.read_csv('test_group_data.csv')

# Compare the distributions of 'Total Score extended' between the two groups using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='Project name', y='Total Score extended', data=pd.concat([control_group_data, test_group_data]))
plt.xlabel('Group')
plt.ylabel('Total Score extended')
plt.title('Comparison of Total Score extended between Groups')
plt.show()

# %%
control_group_data

# %%
test_group_data

# %%
sns.countplot(x='Eye movement type', data=control_group_data)
plt.title("Distribution of Eye Movement Types")
plt.xlabel("Eye Movement Type")
plt.ylabel("Count of Observations")

# Display plot
plt.show()


# %%
sns.countplot(x='Eye movement type', data=test_group_data)
plt.title("Distribution of Eye Movement Types")
plt.xlabel("Eye Movement Type")
plt.ylabel("Count of Observations")

# Display plot
plt.show()


# %%
# Scatter plot to visualize the distribution of gaze points on the screen
plt.scatter(control_group_data['Gaze point X'], control_group_data['Gaze point Y'], alpha=0.5)
plt.xlabel('Gaze Point X (Horizontal Position)')
plt.ylabel('Gaze Point Y (Vertical Position)')
plt.title('Distribution of Control Group Gaze Points on Screen')

plt.show()

# %%
# Scatter plot to visualize the distribution of gaze points on the screen
plt.scatter(test_group_data['Gaze point X'], test_group_data['Gaze point Y'], alpha=0.5)
plt.xlabel('Gaze Point X (Horizontal Position)')
plt.ylabel('Gaze Point Y (Vertical Position)')
plt.title('Distribution of Test Group Gaze Points on Screen')

plt.show()

# %%
# Select only numeric columns for the correlation matrix
numeric_columns = control_group_data.select_dtypes(include=[np.number])
control_corr_matrix = numeric_columns.corr()

# Visualize the correlation matrix for the control group
plt.figure(figsize=(10, 8))
sns.heatmap(control_corr_matrix, cmap='coolwarm')
plt.title('Correlation Matrix - Control Group')
plt.show()

# Repeat the same for the test group
numeric_columns_test = test_group_data.select_dtypes(include=[np.number])
test_corr_matrix = numeric_columns_test.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(test_corr_matrix, cmap='coolwarm')
plt.title('Correlation Matrix - Test Group')
plt.show()


# %%
# Get list of numeric column names from control group data
control_numeric_columns = control_group_data.select_dtypes(include=[np.number]).columns.tolist()

# Get list of numeric column names from test group data
test_numeric_columns = test_group_data.select_dtypes(include=[np.number]).columns.tolist()


# %%
control_group_column_names = control_group_data.columns.tolist()
control_group_column_names

# %% [markdown]
# #  Machine Learning 
# #Model Building

# %%
import pandas as pd


control_group_data = pd.read_csv('control_group_data.csv')
test_group_data = pd.read_csv('test_group_data.csv')

# Select relevant columns for control group
control_selected_columns = ['Participant name', 'Recording duration',
                             'Pupil diameter left', 'Pupil diameter right',
                             'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
                             'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)',
                             'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y',
                             'Gaze event duration', 'Fixation point X', 'Fixation point Y', 'Total Score extended', 'Gaze point X', 'Gaze point Y', 'Gaze event duration']

# Create a DataFrame with selected columns for control group
control_group_selected = control_group_data[control_selected_columns]

# Select relevant columns for test group
test_selected_columns = ['Participant name', 'Recording duration',
                         'Pupil diameter left', 'Pupil diameter right',
                         'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
                         'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)',
                         'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y',
                         'Gaze event duration', 'Fixation point X', 'Fixation point Y', 'Total Score extended', 'Gaze point X', 'Gaze point Y', 'Gaze event duration']

# Create a DataFrame with selected columns for test group
test_group_selected = test_group_data[test_selected_columns]

# %%
# Replace NaN values with 0 in control group dataframe
control_group_selected.fillna(0, inplace=True)

# Replace NaN values with 0 in test group dataframe
test_group_selected.fillna(0, inplace=True)

# %%
# Feature Engineering
# Calculate Eye Movement Ratios for X, Y, and Z
control_group_selected['Eye_Position_Ratio_X'] = control_group_selected['Eye position left X (DACSmm)'] / (control_group_selected['Eye position right X (DACSmm)'] + 1e-6)
test_group_selected['Eye_Position_Ratio_X'] = test_group_selected['Eye position left X (DACSmm)'] / (test_group_selected['Eye position right X (DACSmm)'] + 1e-6)

control_group_selected['Eye_Position_Ratio_Y'] = control_group_selected['Eye position left Y (DACSmm)'] / (control_group_selected['Eye position right Y (DACSmm)'] + 1e-6)
test_group_selected['Eye_Position_Ratio_Y'] = test_group_selected['Eye position left Y (DACSmm)'] / (test_group_selected['Eye position right Y (DACSmm)'] + 1e-6)

control_group_selected['Eye_Position_Ratio_Z'] = control_group_selected['Eye position left Z (DACSmm)'] / (control_group_selected['Eye position right Z (DACSmm)'] + 1e-6)
test_group_selected['Eye_Position_Ratio_Z'] = test_group_selected['Eye position left Z (DACSmm)'] / (test_group_selected['Eye position right Z (DACSmm)'] + 1e-6)


# %%
# Calculate Gaze Point Differences for X and Y
control_group_selected['Gaze_Point_Diff_X'] = control_group_selected['Gaze point left Y'] - control_group_selected['Gaze point right X']
test_group_selected['Gaze_Point_Diff_X'] = test_group_selected['Gaze point left Y'] - test_group_selected['Gaze point right X']

control_group_selected['Gaze_Point_Diff_Y'] = control_group_selected['Gaze point left Y'] - control_group_selected['Gaze point right Y']
test_group_selected['Gaze_Point_Diff_Y'] = test_group_selected['Gaze point left Y'] - test_group_selected['Gaze point right Y']

# %%
# Drop columns used in feature engineering
columns_to_drop = ['Eye position left X (DACSmm)', 'Eye position right X (DACSmm)',
                   'Eye position left Y (DACSmm)', 'Eye position right Y (DACSmm)',
                   'Eye position left Z (DACSmm)', 'Eye position right Z (DACSmm)',
                   'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y',
                   ]

control_group_selected.drop(columns=columns_to_drop, inplace=True)
test_group_selected.drop(columns=columns_to_drop, inplace=True)


# %%
control_group_selected

# %%
test_group_selected

# %% [markdown]
# #Define the input features (X) and target variable (y) for control group

# %%
# Define the input features (X) and target variable (y) for control group
X_control = control_group_selected.drop(columns=['Total Score extended', 'Participant name', 'Recording duration'])
y_control = control_group_selected['Total Score extended']
participants_control = control_group_selected['Participant name']

# Define the input features (X) and target variable (y) for test group
X_test = test_group_selected.drop(columns=['Total Score extended', 'Participant name', 'Recording duration'])
y_test = test_group_selected['Total Score extended']
participants_test = test_group_selected['Participant name']


# %% [markdown]
# #Function to print evaluation metrics

# %%
# Function to print evaluation metrics
def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'{name} - R^2 Score: {r2}')
    print(f'{name} - Mean Squared Error: {mse}')
    print(f'{name} - Mean Absolute Error: {mae}')
    print(f'{name} - Root Mean Squared Error: {rmse}\n')

# %% [markdown]
# # LinearRegression

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Linear Regression model instantiation
lr_model = LinearRegression()

# Training and evaluation for Control Group
lr_model.fit(X_control, y_control)
predictions_control = lr_model.predict(X_control)
print("Linear Regression - Control Group:")
print_metrics("Linear Regression", y_control, predictions_control)

# Training and evaluation for Test Group
lr_model.fit(X_test, y_test)
predictions_test = lr_model.predict(X_test)
print("Linear Regression - Test Group:")
print_metrics("Linear Regression", y_test, predictions_test)



# %%
# Create a single plot for both groups
plt.figure(figsize=(12, 6))

# Plot actual and predicted scores for the control group
plt.scatter(y_control, y_control, alpha=0.5, label='Actual (Control)', color='blue')
plt.scatter(y_control, predictions_control, alpha=0.5, label='Predicted (Control)', color='orange')

# Plot actual and predicted scores for the test group
plt.scatter(y_test, y_test, alpha=0.5, label='Actual (Test)', color='green')
plt.scatter(y_test, predictions_test, alpha=0.5, label='Predicted (Test)', color='purple')

plt.title('Actual vs. Predicted Scores Linear Regression')
plt.xlabel('Actual Score')
plt.ylabel('Predicted/Actual Score')
plt.legend()
plt.plot([min(y_control.min(), y_test.min()), max(y_control.max(), y_test.max())], [min(y_control.min(), y_test.min()), max(y_control.max(), y_test.max())], color='red')  # identity line
plt.show()

# %%
# Calculate Correlation of Actual Scores and Predicted Scores - Control Group
correlation_control_lr = np.corrcoef(y_control, predictions_control)[0, 1]

# Calculate Correlation of Actual Scores and Predicted Scores - Test Group
correlation_test_lr = np.corrcoef(y_test, predictions_test)[0, 1]

# Print Correlation Values for Linear Regression
print(f"Correlation of Actual Scores and Predicted Scores - Control Group (Linear Regression): {correlation_control_lr:.2f}")
print(f"Correlation of Actual Scores and Predicted Scores - Test Group (Linear Regression): {correlation_test_lr:.2f}")

# %%
# Calculate Correlation of Top 3 Features with Actual Scores - Control Group
top3_features_control_lr = X_control.iloc[:, :3]
correlation_top3_control_lr = top3_features_control_lr.corrwith(y_control)

# Calculate Correlation of Top 3 Features with Actual Scores - Test Group
top3_features_test_lr = X_test.iloc[:, :3]
correlation_top3_test_lr = top3_features_test_lr.corrwith(y_test)

# Print Correlation of Top 3 Features for Linear Regression
print("\nCorrelation of Top 3 Features with Actual Scores - Control Group (Linear Regression):")
print(correlation_top3_control_lr)
print("\nCorrelation of Top 3 Features with Actual Scores - Test Group (Linear Regression):")
print(correlation_top3_test_lr)


# %% [markdown]
# # DecisionTreeRegressor

# %%
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42) # You can adjust max_depth

# Training and evaluation for Decision Tree Regressor

print("Decision Tree Regressor - Control Group:")
dt_model.fit(X_control, y_control)
predictions_control = dt_model.predict(X_control)
print_metrics("Decision Tree Regressor", y_control, predictions_control)

print("Decision Tree Regressor - Test Group:")
dt_model.fit(X_test, y_test)
predictions_test = dt_model.predict(X_test)
print_metrics("Decision Tree Regressor", y_test, predictions_test)
print("\n")

# %%
# Create a single plot for Decision Tree Regressor - Both Groups
plt.figure(figsize=(12, 6))
plt.scatter(y_control, predictions_control, alpha=0.5, label='Control Group Predictions', color='orange')
plt.scatter(y_test, predictions_test, alpha=0.5, label='Test Group Predictions', color='purple')
plt.title('Actual vs. Predicted Scores - Decision Tree Regressor')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.plot([min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], [min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], color='red')  # identity line
plt.show()


# %%
# Calculate Correlation of Actual Scores and Predicted Scores - Control Group
correlation_control = np.corrcoef(y_control, predictions_control)[0, 1]

# Calculate Correlation of Actual Scores and Predicted Scores - Test Group
correlation_test = np.corrcoef(y_test, predictions_test)[0, 1]

# Print Correlation Values
print(f"Correlation of Actual Scores and Predicted Scores - Decision Tree Regressor - Control Group: {correlation_control:.2f}")
print(f"Correlation of Actual Scores and Predicted Scores - Decision Tree Regressor - Test Group: {correlation_test:.2f}")


# %%
# Calculate Correlation of Top 3 Features with Actual Scores - Control Group
top3_features_control = X_control.iloc[:, :3]
correlation_top3_control = top3_features_control.corrwith(y_control)

# Calculate Correlation of Top 3 Features with Actual Scores - Test Group
top3_features_test = X_test.iloc[:, :3]
correlation_top3_test = top3_features_test.corrwith(y_test)

# Print Correlation of Top 3 Features
print("\nCorrelation of Top 3 Features with Actual Scores - Decision Tree Regressor - Control Group:")
print(correlation_top3_control)
print("\nCorrelation of Top 3 Features with Actual Scores - Decision Tree Regressor - Test Group:")
print(correlation_top3_test)


# %% [markdown]
# # ElasticNet Regression

# %%
from sklearn.linear_model import ElasticNet
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)  

# Training and evaluation for ElasticNet
print("ElasticNet - Control Group:")
elasticnet_model.fit(X_control, y_control)
predictions_control = elasticnet_model.predict(X_control)
print_metrics("ElasticNet", y_control, predictions_control)

print("ElasticNet - Test Group:")
elasticnet_model.fit(X_test, y_test)
predictions_test = elasticnet_model.predict(X_test)
print_metrics("ElasticNet", y_test, predictions_test)
print("\n")


# %%
# Create a single plot for ElasticNet Regression - Both Groups
plt.figure(figsize=(12, 6))
plt.scatter(y_control, predictions_control, alpha=0.5, label='Control Group Predictions', color='orange')
plt.scatter(y_test, predictions_test, alpha=0.5, label='Test Group Predictions', color='purple')
plt.title('Actual vs. Predicted Scores - ElasticNet Regression')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.plot([min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], [min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], color='red')  # identity line
plt.show()

# %%
# Calculate Correlation of Actual Scores and Predicted Scores - Control Group
correlation_control_elasticnet = np.corrcoef(y_control, predictions_control)[0, 1]

# Calculate Correlation of Actual Scores and Predicted Scores - Test Group
correlation_test_elasticnet = np.corrcoef(y_test, predictions_test)[0, 1]

# Print Correlation Values for ElasticNet Regression
print(f"Correlation of Actual Scores and Predicted Scores - ElasticNet (Control Group): {correlation_control_elasticnet:.2f}")
print(f"Correlation of Actual Scores and Predicted Scores - ElasticNet (Test Group): {correlation_test_elasticnet:.2f}")

# %%
# Calculate Correlation of Top 3 Features with Actual Scores - Control Group
top3_features_control = X_control.iloc[:, :3]
correlation_top3_control_elasticnet = top3_features_control.corrwith(y_control)

# Calculate Correlation of Top 3 Features with Actual Scores - Test Group
top3_features_test = X_test.iloc[:, :3]
correlation_top3_test_elasticnet = top3_features_test.corrwith(y_test)

# Print Correlation of Top 3 Features for ElasticNet Regression
print("\nCorrelation of Top 3 Features with Actual Scores - Control Group (ElasticNet):")
print(correlation_top3_control_elasticnet)
print("\nCorrelation of Top 3 Features with Actual Scores - Test Group (ElasticNet):")
print(correlation_top3_test_elasticnet)


# %% [markdown]
# # Gradient Boosting Regressor

# %%
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Training and evaluation for Gradient Boosting Regressor
print("Gradient Boosting Regressor - Control Group:")
gb_model.fit(X_control, y_control)
predictions_control = gb_model.predict(X_control)
print_metrics("Gradient Boosting Regressor", y_control, predictions_control)

print("Gradient Boosting Regressor - Test Group:")
gb_model.fit(X_test, y_test)
predictions_test = gb_model.predict(X_test)
print_metrics("Gradient Boosting Regressor", y_test, predictions_test)
print("\n")


# %%
plt.figure(figsize=(12, 6))
plt.scatter(y_control, predictions_control, alpha=0.5, label='Control Group Predictions', color='orange')
plt.scatter(y_test, predictions_test, alpha=0.5, label='Test Group Predictions', color='purple')
plt.title('Actual vs. Predicted Scores - Gradient Boosting Regressor')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.plot([min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], [min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], color='red')  # identity line
plt.show()


# %%
# Calculate Correlation of Actual Scores and Predicted Scores - Control Group
correlation_control_gb = np.corrcoef(y_control, predictions_control)[0, 1]

# Calculate Correlation of Actual Scores and Predicted Scores - Test Group
correlation_test_gb = np.corrcoef(y_test, predictions_test)[0, 1]

# Print Correlation Values for Gradient Boosting Regressor
print(f"Correlation of Actual Scores and Predicted Scores - Gradient Boosting (Control Group): {correlation_control_gb:.2f}")
print(f"Correlation of Actual Scores and Predicted Scores - Gradient Boosting (Test Group): {correlation_test_gb:.2f}")

# %%
# Calculate Correlation of Top 3 Features with Actual Scores - Control Group
top3_features_control_gb = X_control.iloc[:, :3]
correlation_top3_control_gb = top3_features_control_gb.corrwith(y_control)

# Calculate Correlation of Top 3 Features with Actual Scores - Test Group
top3_features_test_gb = X_test.iloc[:, :3]
correlation_top3_test_gb = top3_features_test_gb.corrwith(y_test)

# Print Correlation of Top 3 Features for Gradient Boosting Regressor
print("\nCorrelation of Top 3 Features with Actual Scores - Control Group (Gradient Boosting):")
print(correlation_top3_control_gb)
print("\nCorrelation of Top 3 Features with Actual Scores - Test Group (Gradient Boosting):")
print(correlation_top3_test_gb)


# %% [markdown]
# # RandomForestRegressor
# 

# %%
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training and evaluation for Random Forest Regressor
print("Random Forest Regressor - Control Group:")
rf_model.fit(X_control, y_control)
predictions_control_rf = rf_model.predict(X_control)
print_metrics("Random Forest Regressor", y_control, predictions_control_rf)

print("Random Forest Regressor - Test Group:")
rf_model.fit(X_test, y_test)
predictions_test_rf = rf_model.predict(X_test)
print_metrics("Random Forest Regressor", y_test, predictions_test_rf)


# %%
plt.figure(figsize=(12, 6))
plt.scatter(y_control, predictions_control_rf, alpha=0.5, label='Control Group Predictions', color='orange')
plt.scatter(y_test, predictions_test_rf, alpha=0.5, label='Test Group Predictions', color='purple')
plt.title('Actual vs. Predicted Scores - Random Forest Regressor')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.plot([min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], [min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], color='red')  # identity line
plt.show()


# %%
# Calculate Correlation of Actual Scores and Predicted Scores - Control Group
correlation_control_rf = np.corrcoef(y_control, predictions_control_rf)[0, 1]

# Calculate Correlation of Actual Scores and Predicted Scores - Test Group
correlation_test_rf = np.corrcoef(y_test, predictions_test_rf)[0, 1]

# Print Correlation Values for Random Forest Regressor
print(f"Correlation of Actual Scores and Predicted Scores - Random Forest (Control Group): {correlation_control_rf:.2f}")
print(f"Correlation of Actual Scores and Predicted Scores - Random Forest (Test Group): {correlation_test_rf:.2f}")


# %%
# Calculate Correlation of Top 3 Features with Actual Scores - Control Group
top3_features_control = X_control.iloc[:, :3]
correlation_top3_control_rf = top3_features_control.corrwith(y_control)

# Calculate Correlation of Top 3 Features with Actual Scores - Test Group
top3_features_test = X_test.iloc[:, :3]
correlation_top3_test_rf = top3_features_test.corrwith(y_test)

# Print Correlation of Top 3 Features for Random Forest Regressor
print("\nCorrelation of Top 3 Features with Actual Scores - Control Group (Random Forest):")
print(correlation_top3_control_rf)
print("\nCorrelation of Top 3 Features with Actual Scores - Test Group (Random Forest):")
print(correlation_top3_test_rf)


# %% [markdown]
# # K-Nearest Neighbors Regression

# %%
from sklearn.neighbors import KNeighborsRegressor

# Create a K-Nearest Neighbors Regressor model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Training and evaluation for K-Nearest Neighbors Regressor
print("K-Nearest Neighbors Regressor - Control Group:")
knn_model.fit(X_control, y_control)
predictions_control_knn = knn_model.predict(X_control)
print_metrics("K-Nearest Neighbors Regressor", y_control, predictions_control_knn)

print("K-Nearest Neighbors Regressor - Test Group:")
knn_model.fit(X_test, y_test)
predictions_test_knn = knn_model.predict(X_test)
print_metrics("K-Nearest Neighbors Regressor", y_test, predictions_test_knn)


# %%
# Calculate Correlation of Actual Scores and Predicted Scores - Control Group
correlation_control_knn = np.corrcoef(y_control, predictions_control_knn)[0, 1]

# Calculate Correlation of Actual Scores and Predicted Scores - Test Group
correlation_test_knn = np.corrcoef(y_test, predictions_test_knn)[0, 1]

# Print Correlation Values for K-Nearest Neighbors Regressor
print(f"Correlation of Actual Scores and Predicted Scores - K-Nearest Neighbors (Control Group): {correlation_control_knn:.2f}")
print(f"Correlation of Actual Scores and Predicted Scores - K-Nearest Neighbors (Test Group): {correlation_test_knn:.2f}")


# %%
plt.figure(figsize=(12, 6))
plt.scatter(y_control, predictions_control_knn, alpha=0.5, label='Control Group Predictions', color='orange')
plt.scatter(y_test, predictions_test_knn, alpha=0.5, label='Test Group Predictions', color='purple')
plt.title('Actual vs. Predicted Scores - K-Nearest Neighbors Regressor')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.plot([min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], [min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], color='red')  # identity line
plt.show()


# %%
# Calculate Correlation of Top 3 Features with Actual Scores - Control Group
top3_features_control = X_control.iloc[:, :3]
correlation_top3_control_knn = top3_features_control.corrwith(y_control)

# Calculate Correlation of Top 3 Features with Actual Scores - Test Group
top3_features_test = X_test.iloc[:, :3]
correlation_top3_test_knn = top3_features_test.corrwith(y_test)

# Print Correlation of Top 3 Features for K-Nearest Neighbors Regressor
print("\nCorrelation of Top 3 Features with Actual Scores - Control Group (KNN):")
print(correlation_top3_control_knn)
print("\nCorrelation of Top 3 Features with Actual Scores - Test Group (KNN):")
print(correlation_top3_test_knn)


# %% [markdown]
# # Neural Networks

# %%
from sklearn.neural_network import MLPRegressor

# Create a Neural Network Regressor model
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)

# Training and evaluation for Neural Network Regressor - Control Group
print("Neural Network Regressor - Control Group:")
nn_model.fit(X_control, y_control)
predictions_control_nn = nn_model.predict(X_control)
print_metrics("Neural Network Regressor", y_control, predictions_control_nn)

# Training and evaluation for Neural Network Regressor - Test Group
print("Neural Network Regressor - Test Group:")
nn_model.fit(X_test, y_test)
predictions_test_nn = nn_model.predict(X_test)
print_metrics("Neural Network Regressor", y_test, predictions_test_nn)


# %%
# Calculate Correlation of Actual Scores and Predicted Scores - Control Group
correlation_control_knn = np.corrcoef(y_control, predictions_control_nn)[0, 1]

# Calculate Correlation of Actual Scores and Predicted Scores - Test Group
correlation_test_knn = np.corrcoef(y_test, predictions_test_nn)[0, 1]

# Print Correlation Values for K-Nearest Neighbors Regressor
print(f"Correlation of Actual Scores and Predicted Scores - Neural Network Regressor (Control Group): {correlation_control_knn:.2f}")
print(f"Correlation of Actual Scores and Predicted Scores -Neural Network Regressor (Test Group): {correlation_test_knn:.2f}")


# %%
# Create a single plot for Neural Network Regressor - Both Groups
plt.figure(figsize=(12, 6))
plt.scatter(y_control, predictions_control_nn, alpha=0.5, label='Control Group Predictions', color='orange')
plt.scatter(y_test, predictions_test_nn, alpha=0.5, label='Test Group Predictions', color='purple')
plt.title('Actual vs. Predicted Scores - Neural Network Regressor')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.plot([min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], [min(min(y_control), min(y_test)), max(max(y_control), max(y_test))], color='red')  # identity line
plt.show()

# %%
# Calculate Correlation of Top 3 Features with Actual Scores - Control Group
top3_features_control = X_control.iloc[:, :3]
correlation_top3_control_nn = top3_features_control.corrwith(y_control)

# Calculate Correlation of Top 3 Features with Actual Scores - Test Group
top3_features_test = X_test.iloc[:, :3]
correlation_top3_test_nn = top3_features_test.corrwith(y_test)

# Print Correlation of Top 3 Features for Neural Network Regressor
print("\nCorrelation of Top 3 Features with Actual Scores - Control Group (Neural Network):")
print(correlation_top3_control_nn)
print("\nCorrelation of Top 3 Features with Actual Scores - Test Group (Neural Network):")
print(correlation_top3_test_nn)

# %% [markdown]
# 
# # Unsupervised Learning

# %% [markdown]
# Performing K-means Clustering.

# %%
# Select the columns to be used in the clustering analysis
X = data[['Gaze point X', 'Gaze point Y', 'Gaze point left X', 'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y']]

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# %%
# Perform k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original dataframe
data['cluster'] = y_kmeans

# Plot the clusters in 2D space
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters')
plt.xlabel('Gaze point X')
plt.ylabel('Gaze point Y')
plt.legend()
plt.show()

# Calculate the average gaze duration for each cluster
empathy_scores = data['Gaze event duration'].unique()
for i in range(3):
    cluster_mean = []
    for score in empathy_scores:
        cluster_mean.append(data[data['cluster'] == i][data['Gaze event duration'] == score]['Gaze event duration'].count())
    print('Cluster', i+1, 'Gaze event duration score:', sum(cluster_mean) / len(cluster_mean))

# %%
data = data.drop('cluster', axis=1)

# %% [markdown]
# # Modelling of the data

# %%
from sklearn.preprocessing import LabelEncoder

# select categorical columns
cat_cols = data.select_dtypes(include='object').columns.tolist()

# initialize label encoder object
label_encoder = LabelEncoder()

# encode categorical columns
for col in cat_cols:
    data[col] = label_encoder.fit_transform(data[col])

# %%
new_data_b = pd.concat([data, new_data_a['Gaze event duration']], axis=1)

# %%
X = new_data_b.drop('Gaze event duration',axis=1)
y = new_data_b['Gaze event duration']

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


# split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# define early stopping criteria
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)


# compile the model
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['mae', 'mse'])

# train the model
history = model.fit(X_train_scaled, y_train, epochs=5, validation_data=(X_val_scaled, y_val), batch_size=128, callbacks=[early_stop])

# evaluate the model on the test set
results = model.evaluate(X_test_scaled, y_test)
print("test loss, test mae, test mse:", results)


# %%
# Get the MAE and MSE values from the history object
mae = history.history['mae']
mse = history.history['mse']

# Plot the MAE and MSE curves
plt.plot(mae, label='MAE')
plt.plot(mse, label='MSE')
plt.title('Training MAE and MSE')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# %%
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# %% [markdown]
# # Lasso and Ridge Regression.

# %%
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

# %%
# Create Lasso regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Make predictions on test data using Lasso model
y_pred_lasso = lasso_model.predict(X_test)

# Calculate Lasso regression metrics
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
# calculating MAE for Lasso regression
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print("Lasso Regression metrics:")
print("Mean Squared Error: ", mse_lasso)
print("R^2 Score: ", r2_lasso)
print("Mean absolute Error:",mae_lasso)

# Create Ridge regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Make predictions on test data using Ridge model
y_pred_ridge = ridge_model.predict(X_test)

# Calculate Ridge regression metrics
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
# calculating MAE for Ridge regression
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)


print("Ridge Regression metrics:")
print("Mean Squared Error: ", mse_ridge)
print("R^2 Score: ", r2_ridge)
print("Mean absolute Error:",mae_ridge)

# %% [markdown]
# # For Classification 

# %%
new_data_b.columns = [col.strip() for col in new_data_b.columns]
print (new_data_b.columns)

# %%
class_X = new_data_b.drop('Eye movement type index',axis=1)
class_y = new_data_b['Eye movement type index']

# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#splitting dataset again into the train , validation and test set.
X_train, X_test, y_train, y_test = train_test_split(class_X, class_y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)


# reshape input data for GRU layer
X_train_gru = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_val_gru = np.reshape(X_val, (X_val.shape[0], X_val.shape[1]))
X_test_gru = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))


# Define the model architecture
model = Sequential()

# Add GRU layer
model.add(GRU(64, activation='relu', input_shape=(X_train.shape[1], 1)))


# Add dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Add dense output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set early stopping criteria to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(X_train_gru, y_train, epochs=10, batch_size=64, validation_data=(X_val_gru, y_val), callbacks=[early_stopping])

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_gru, y_test)

# Make predictions on test data
y_pred = model.predict(X_test_gru)


# %% [markdown]
# # Model Analysis

# %%
import pandas as pd
import matplotlib.pyplot as plt

models = ['Linear Regression', 'Decision Tree', 'ElasticNet', 'Gradient Boosting', 
          'Random Forest', 'K-Nearest Neighbors', 'Neural Networks']

# Metrics for Control Group
r2_control = [0.0859, 0.3977, 0.0839, 0.6310, 0.9194, 0.8190, -7944405.0750]
mse_control = [56.2487, 37.0648, 56.3727, 22.7064, 4.9613, 11.1396, 488878579.9099]
mae_control = [5.1457, 3.0228, 5.1539, 2.4114, 0.4500, 0.7076, 7816.8776]
rmse_control = [7.4999, 6.0881, 7.5082, 4.7651, 2.2274, 3.3376, 22110.5988]

# Metrics for Test Group
r2_test = [0.0789, 0.3321, 0.0782, 0.6518, 0.9163, 0.8560, 0.6154]
mse_test = [235.7893, 170.9598, 235.9669, 89.1305, 21.4378, 36.8649, 98.4577]
mae_test = [12.2950, 8.3633, 12.2838, 6.0279, 1.0039, 1.5130, 5.9814]
rmse_test = [15.3554, 13.0752, 15.3612, 9.4409, 4.6301, 6.0716, 9.9226]
correlation_control = [0.80, 0.80, 0.29, 0.80, 0.96, 0.91, -0.04]
correlation_test = [0.81, 0.81, 0.28, 0.81, 0.96, 0.93, 0.79]

# Creating the DataFrame
data = {
    'Model': models,
    'R^2 Score (Control)': r2_control,
    'MSE (Control)': mse_control,
    'MAE (Control)': mae_control,
    'RMSE (Control)': rmse_control,
    'R^2 Score (Test)': r2_test,
    'MSE (Test)': mse_test,
    'MAE (Test)': mae_test,
    'RMSE (Test)': rmse_test,
    'Correlation_Control' : correlation_control,
    'Correlation_Test' : correlation_test

}

df = pd.DataFrame(data)
print(df)

# Transposing data for correct orientation in table
table_data = list(map(list, zip(*data.values())))

# Create table
fig, ax = plt.subplots(figsize=(18, 8))  # set the size that you'd like (width, height)
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_data,
         colLabels=list(data.keys()),  # Convert dict_keys to list
         cellLoc='center',
         loc='center')

plt.show()



# %%
# Control Group Data
data_control = {
    'Model': models,
    'R^2 Score (Control)': r2_control,
    'MSE (Control)': mse_control,
    'MAE (Control)': mae_control,
    'RMSE (Control)': rmse_control,
    'Correlation_Control' : correlation_control
}

# Test Group Data
data_test = {
    'Model': models,
    'R^2 Score (Test)': r2_test,
    'MSE (Test)': mse_test,
    'MAE (Test)': mae_test,
    'RMSE (Test)': rmse_test,
    'Correlation_Test' : correlation_test
}

# Transposing data for correct orientation in table
table_data_control = list(map(list, zip(*data_control.values())))
table_data_test = list(map(list, zip(*data_test.values())))

# Create Control Group table
fig_control, ax_control = plt.subplots(figsize=(18, 8))  # set the size that you'd like (width, height)
ax_control.axis('tight')
ax_control.axis('off')
ax_control.set_title('Control Group Metrics')
ax_control.table(cellText=table_data_control,
                 colLabels=list(data_control.keys()),
                 cellLoc='center',
                 loc='center')

# Create Test Group table
fig_test, ax_test = plt.subplots(figsize=(18, 8))  # set the size that you'd like (width, height)
ax_test.axis('tight')
ax_test.axis('off')
ax_test.set_title('Test Group Metrics')
ax_test.table(cellText=table_data_test,
              colLabels=list(data_test.keys()),
              cellLoc='center',
              loc='center')

plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Define the index for x-axis placement
index = np.arange(len(models))

# Plotting R^2 Score
plt.figure(figsize=(15, 6))
plt.bar(index, r2_control, width=0.4, label='Control Group', color='b', align='center')
plt.bar(index + 0.4, r2_test, width=0.4, label='Test Group', color='r', align='center')
plt.xlabel('Models')
plt.ylabel('R^2 Score')
plt.title('R^2 Score by Model')
plt.xticks(index + 0.2, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting MSE
plt.figure(figsize=(15, 6))
plt.bar(index, mse_control, width=0.4, label='Control Group', color='b', align='center')
plt.bar(index + 0.4, mse_test, width=0.4, label='Test Group', color='r', align='center')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE by Model')
plt.xticks(index + 0.2, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting MAE
plt.figure(figsize=(15, 6))
plt.bar(index, mae_control, width=0.4, label='Control Group', color='b', align='center')
plt.bar(index + 0.4, mae_test, width=0.4, label='Test Group', color='r', align='center')
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE by Model')
plt.xticks(index + 0.2, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting RMSE
plt.figure(figsize=(15, 6))
plt.bar(index, rmse_control, width=0.4, label='Control Group', color='b', align='center')
plt.bar(index + 0.4, rmse_test, width=0.4, label='Test Group', color='r', align='center')
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE by Model')
plt.xticks(index + 0.2, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting Correlation Scores
plt.figure(figsize=(15, 6))
plt.bar(index, correlation_control, width=0.4, label='Control Group', color='b', align='center')
plt.bar(index + 0.4, correlation_test, width=0.4, label='Test Group', color='r', align='center')
plt.xlabel('Models')
plt.ylabel('Correlation')
plt.title('Correlation by Model')
plt.xticks(index + 0.2, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

models = ['Linear Regression', 'Decision Tree', 'ElasticNet', 'Gradient Boosting', 'Random Forest', 'K-Nearest Neighbors', 'Neural Networks']

# Metrics for Control Group
r2_control = [0.0859, 0.3977, 0.0839, 0.6310, 0.9194, 0.8190, -7944405.0750]
mse_control = [56.2487, 37.0648, 56.3727, 22.7064, 4.9613, 11.1396, 488878579.9099]
mae_control = [5.1457, 3.0228, 5.1539, 2.4114, 0.4500, 0.7076, 7816.8776]
rmse_control = [7.4999, 6.0881, 7.5082, 4.7651, 2.2274, 3.3376, 22110.5988]

# Metrics for Test Group
r2_test = [0.0789, 0.3321, 0.0782, 0.6518, 0.9163, 0.8560, 0.6154]
mse_test = [235.7893, 170.9598, 235.9669, 89.1305, 21.4378, 36.8649, 98.4577]
mae_test = [12.2950, 8.3633, 12.2838, 6.0279, 1.0039, 1.5130, 5.9814]
rmse_test = [15.3554, 13.0752, 15.3612, 9.4409, 4.6301, 6.0716, 9.9226]

index = np.arange(len(models))

# Plotting for Control Group
fig, axs = plt.subplots(4, 2, figsize=(20, 28))
axs[0, 0].bar(index, r2_control, color='b', width=0.4)
axs[0, 0].set_title('R^2 Scores for Control Group')
axs[0, 0].set_xticks(index)
axs[0, 0].set_xticklabels(models, rotation=45, ha='right')

axs[1, 0].bar(index, mse_control, color='b', width=0.4)
axs[1, 0].set_title('MSE for Control Group')
axs[1, 0].set_xticks(index)
axs[1, 0].set_xticklabels(models, rotation=45, ha='right')

axs[2, 0].bar(index, mae_control, color='b', width=0.4)
axs[2, 0].set_title('MAE for Control Group')
axs[2, 0].set_xticks(index)
axs[2, 0].set_xticklabels(models, rotation=45, ha='right')

axs[3, 0].bar(index, rmse_control, color='b', width=0.4)
axs[3, 0].set_title('RMSE for Control Group')
axs[3, 0].set_xticks(index)
axs[3, 0].set_xticklabels(models, rotation=45, ha='right')

# Plotting for Test Group
axs[0, 1].bar(index, r2_test, color='r', width=0.4)
axs[0, 1].set_title('R^2 Scores for Test Group')
axs[0, 1].set_xticks(index)
axs[0, 1].set_xticklabels(models, rotation=45, ha='right')

axs[1, 1].bar(index, mse_test, color='r', width=0.4)
axs[1, 1].set_title('MSE for Test Group')
axs[1, 1].set_xticks(index)
axs[1, 1].set_xticklabels(models, rotation=45, ha='right')

axs[2, 1].bar(index, mae_test, color='r', width=0.4)
axs[2, 1].set_title('MAE for Test Group')
axs[2, 1].set_xticks(index)
axs[2, 1].set_xticklabels(models, rotation=45, ha='right')

axs[3, 1].bar(index, rmse_test, color='r', width=0.4)
axs[3, 1].set_title('RMSE for Test Group')
axs[3, 1].set_xticks(index)
axs[3, 1].set_xticklabels(models, rotation=45, ha='right')

plt.tight_layout()
plt.show()


# %% [markdown]
# # Conclusion

# %% [markdown]
# Our study reveals that gaze dynamics hold promise as indicators of empathy levels. Machine learn-
# ing models, notably Random Forest Regressor and Gradient Boosting Regressor, demonstrated
# strong predictive capabilities, emphasizing the potential of gaze-related features for empathy as-
# sessment. The consistent significance of pupil diameters and gaze event duration across models
# and groups highlights their pivotal role. These findings open avenues for advancing empathy
# measurement, bridging psychology and technology.
# 


