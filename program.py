import pandas as pd
import streamlit as st


st.title('Student Success Analytics')

uploaded_file = st.file_uploader('Upload your file here')

if uploaded_file:
  data = pd.read_csv('student_performance_updated_1000.csv')
  st.write(data.describe()) 

  st.write('Exploratory Data Analysis')

  st.write('Column names: ', data.columns)

  st.write('Shape: ', data.shape)

  st.write('Data types: ', data.dtypes)

  st.write('Missing values: ', data.isnull().sum())

  st.write('Duplicate rows: ', data[data.duplicated()])

  data = data.dropna()

  data = data.drop_duplicates()

  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.histplot(data['Study Hours'], bins=20, kde=True)
  plt.title('Distribution of study hours per week.')
  plt.show()

  st.write(data.columns.tolist())

  fig, ax = plt.subplots()

  ax.hist(data['CurrentGrade'], bins=20, edgecolor='black')
  ax.set_xlabel('Current Grades')
  ax.set_ylabel('Frequency')
  ax.set_title('Distribution of current grades')

  st.pyplot(fig)

  sns.scatterplot(x='Study Hours', y='CurrentGrade', data=data)
  plt.title('Study hours vs. final grades')
  plt.show()

  correlation_matrix = data.select_dtypes(include='number').corr()

  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Correlation matrix')
  plt.show()

  sns.histplot(data['Attendance (%)'], bins=20, kde=True)
  plt.title('Attendance percentage distribution')
  plt.show()

  X = data['Study Hours']
  y = data['Exam Scores']

  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  from sklearn.linear_model import LinearRegression

  model = LinearRegression()

  X_train = pd.DataFrame(X_train)

  model.fit(X_train, y_train)

  plt.scatter(X_test, y_test, color='blue')
  plt.plot(X_train, model.predict(X_train), color='red')
  plt.title('Linear Regression Visualization')
  plt.xlabel('Study Hours')
  plt.ylabel('Exam Grades')
  plt.show()
