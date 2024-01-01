# Hybrid Recommender System

## Business Problem

Make 10 movie recommendations for the user whose ID is given, using the item-based and user-based recommender methods.

<p align="center" width="100%">
    <img width="60%" src="recommender_systems.jpg">
</p>

## Dataset

The dataset was provided by MovieLens, a movie recommendation service. It contains the movies as well as the rating points made for these movies. Contains 2.000.0263 ratings across 27.278 movies. This data set was created on October 17, 2016. It includes 138,493 users and data between January 9, 1995 and March 31, 2015. Users were randomly selected. It is known that all selected users voted for at least 20 movies.

movie.csv
| Column    | Description                |
|-----------|----------------------------|
| movieId   | Unique movie id            |
| title     | Movie name                 |
| genres    | Genre                      |


rating.csv
| Column    | Description                |
|-----------|----------------------------|
| userid    | Unique user id (UniqueID)  |
| movieId   | Unique movie id (UniqueID) |
| rating    | Movie Rating by the user   |
| timestamp | Rating date                |



