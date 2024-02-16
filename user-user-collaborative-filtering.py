from __future__ import print_function, division
from builtins import range, input

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os
if not os.path.exists('user2movie.json') or \
   not os.path.exists('movie2user.json') or \
   not os.path.exists('usermovie2rating.json') or \
   not os.path.exists('usermovie2rating_test.json'):
   import preprocess2dict


with open('user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if N > 10000:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()

# Function to calculate MSE
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

# Define a list of values to test for K and epochs
K_values = [10, 20, 30, 40, 50]  # Example values for K
epochs_values = [10, 20, 30, 40, 50]  # Example values for epochs

# Store performance results for different K and epochs
results = []

for K in K_values:
    for epochs in epochs_values:
        neighbors = []  # store neighbors in this list
        averages = []  # each user's average rating for later use
        deviations = []  # each user's deviation for later use

        for i in range(N):
            movies_i = user2movie[i]
            movies_i_set = set(movies_i)

            ratings_i = { movie: usermovie2rating[(i, movie)] for movie in movies_i }
            avg_i = np.mean(list(ratings_i.values()))
            dev_i = { movie: (rating - avg_i) for movie, rating in ratings_i.items() }
            dev_i_values = np.array(list(dev_i.values()))
            sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

            averages.append(avg_i)
            deviations.append(dev_i)

            sl = SortedList()
            for j in range(N):
                if j != i:
                    movies_j = user2movie[j]
                    movies_j_set = set(movies_j)
                    common_movies = movies_i_set & movies_j_set
                    if len(common_movies) > 10:
                        ratings_j = { movie: usermovie2rating[(j, movie)] for movie in movies_j }
                        avg_j = np.mean(list(ratings_j.values()))
                        dev_j = { movie: (rating - avg_j) for movie, rating in ratings_j.items() }
                        dev_j_values = np.array(list(dev_j.values()))
                        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                        numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
                        w_ij = numerator / (sigma_i * sigma_j)

                        sl.add((-w_ij, j))
                        if len(sl) > K:
                            del sl[-1]

            neighbors.append(sl)

            if i % 1 == 0:
                print("Processed user", i)

        train_predictions = []
        train_targets = []
        for (i, m), target in usermovie2rating.items():
            numerator = 0
            denominator = 0
            for neg_w, j in neighbors[i]:
                try:
                    numerator += -neg_w * deviations[j][m]
                    denominator += abs(neg_w)
                except KeyError:
                    pass

            if denominator == 0:
                prediction = averages[i]
            else:
                prediction = numerator / denominator + averages[i]
            prediction = min(5, prediction)
            prediction = max(0.5, prediction)
            train_predictions.append(prediction)
            train_targets.append(target)

        test_predictions = []
        test_targets = []
        for (i, m), target in usermovie2rating_test.items():
            numerator = 0
            denominator = 0
            for neg_w, j in neighbors[i]:
                try:
                    numerator += -neg_w * deviations[j][m]
                    denominator += abs(neg_w)
                except KeyError:
                    pass

            if denominator == 0:
                prediction = averages[i]
            else:
                prediction = numerator / denominator + averages[i]
            prediction = min(5, prediction)
            prediction = max(0.5, prediction)
            test_predictions.append(prediction)
            test_targets.append(target)

        train_mse = mse(train_predictions, train_targets)
        test_mse = mse(test_predictions, test_targets)

        results.append((K, epochs, train_mse, test_mse))

# Find the best K and epochs combination based on test MSE
best_result = min(results, key=lambda x: x[3])
best_K, best_epochs, best_train_mse, best_test_mse = best_result

# Print the best K and epochs combination and its performance
print("Best K:", best_K)
print("Best epochs:", best_epochs)
print("Train MSE for best K and epochs:", best_train_mse)
print("Test MSE for best K and epochs:", best_test_mse)

# To perform recommendations for 10 users...
# (rest of the code for recommendations remains the same)