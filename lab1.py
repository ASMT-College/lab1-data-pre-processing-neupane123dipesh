#task 1 
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('lab1-datasets/employee_data.csv')
print("Initial Data:\n", df)

# Step 2: Handle missing values (cleaner, no warnings)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# Step 3: Standardize department names
df['Department'] = df['Department'].replace({
    'Human Resources': 'HR',
    'H.R.': 'HR',
    'hr': 'HR'
})

# Step 4: Remove duplicate records based on ID
df = df.drop_duplicates(subset='ID', keep='first')

# Final output
print("\nCleaned Data:\n", df)
'''

#task 2 normalization 

'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('lab1-datasets/student_scores.csv')
print("Original Scores:\n", df)

scaler = MinMaxScaler()
df[['Math', 'Science', 'English']] = scaler.fit_transform(df[['Math', 'Science', 'English']])

print("\nNormalized Scores:\n", df)

'''
#task 3 
'''
import pandas as pd

df = pd.read_csv('lab1-datasets/customer_ages.csv')
print("Original Ages:\n", df)

bins = [18, 30, 50, 100]
labels = ['Young', 'Middle-aged', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

print("\nBinned Data:\n", df)
print("\nGroup Counts:\n", df['AgeGroup'].value_counts())
'''
#task 4
'''
import pandas as pd

df = pd.read_csv('lab1-datasets/sales_data.csv')
print("Original Sales:\n", df)

bins = [0, 5000, 20000, float('inf')]
labels = ['Low', 'Medium', 'High']
df['SalesCategory'] = pd.cut(df['Sales'], bins=bins, labels=labels)

print("\nDiscretized Sales:\n", df)
print("\nSales Category Counts:\n", df['SalesCategory'].value_counts())


#task 5
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Step 1: Load the dataset
df = pd.read_csv('lab1-datasets/medical_data.csv')
print("Initial Data:\n", df.head())

# Step 2: Define features and target variable
# Exclude 'PatientID' because it's just an identifier
X = df.drop(columns=['PatientID', 'Disease'])
y = df['Disease']

# Step 3: Apply Chi-square feature selection
selector = SelectKBest(score_func=chi2, k=3)
selector.fit(X, y)

# Step 4: Get the top 3 features
top_features = X.columns[selector.get_support()]
print("\nTop 3 Features for Predicting Disease:\n", top_features)


