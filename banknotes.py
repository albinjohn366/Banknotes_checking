from sklearn.neighbors import KNeighborsClassifier
import csv
import random

# Creating data for training
with open('banknotes.csv') as file:
    contents = csv.reader(file)
    next(contents)

    data = []
    for row in contents:
        data.append({'evidence': [float(cell) for cell in row[:4]],
                     'label': 'Authentic' if row[4] == '0' else 'Counterfeit'})

# Creating the model
model = KNeighborsClassifier(n_neighbors=3)

# Separating the model for testing and training
holdout = int(0.5 * len(data))
random.shuffle(data)
x_training = [cell['evidence'] for cell in data[:holdout]]
y_training = [cell['label'] for cell in data[:holdout]]
x_testing = [cell['evidence'] for cell in data[holdout:]]
y_testing = [cell['label'] for cell in data[holdout:]]

# Fitting in the model
model.fit(x_training, y_training)

# Making predictions for the testing set
predictions = model.predict(x_testing)

# Checking incorrect and correct model
correct = 0
incorrect = 0
total = 0
for actual, prediction in zip(y_testing, predictions):
    total += 1
    if actual == prediction:
        correct += 1
    else:
        incorrect += 1

# Printing the number of correct and incorrect answers
print('The number of correct prediction is {} which is {:.2f} '
      'percentage'.format(
        correct, correct / total * 100))
print('The number of incorrect prediction is {} which is {:.2f} '
      'percentage'.format(
        incorrect, incorrect / total * 100))
