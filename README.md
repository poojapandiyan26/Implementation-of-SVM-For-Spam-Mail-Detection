# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program.
## Program:

Program to implement the SVM For Spam Mail Detection..
Developed by: V Mythili
RegisterNumber: 212223040123

```
import chardet, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect encoding
with open('spam.csv', 'rb') as f:
    print(chardet.detect(f.read(100000)))
```
```
# Load data
data = pd.read_csv('spam.csv', encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())
```

```
# Split data
x = data['v1'].values   # Labels
y = data['v2'].values   # Messages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Text vectorization
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
```

```
# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```


## Output:

![image](https://github.com/user-attachments/assets/52c8e036-b66d-4eb8-987c-3e382a47653c)

![image](https://github.com/user-attachments/assets/b956fd3f-2066-41b4-b159-ed3622aa3c77)




![image](https://github.com/user-attachments/assets/85b815ab-ad1e-425d-b6c2-f47c6a86f5b0)



![image](https://github.com/user-attachments/assets/abac8205-3b43-42cf-9e48-eeb47e645f2d)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
