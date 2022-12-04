# Predicting the outcome of NBA games
Final project as part of the Lighthouse Labs Data Science Bootcamp. Data used in this project was obtained from [Basketball Reference](https://www.basketball-reference.com/). Work on this project is still ongoing. 

# Goals
 1. Build a machine learning model capable of predicting the outcome of NBA matches. 
 2. Create a model capable of predicting the win percentage of a given team based on advanced team statistics.

# Introduction and Motivation
The NBA is the main basketball league around the world. The league consists of 30 teams divided into two conferences, and a team plays every other team at least twice. Overall, each team plays a total of 82 games over the course of the regular season. 

A lot of data can be generated during NBA games, for example player and team statistics. This data can be used in several different areas, from player performance analysis to devising a team's strategy against specific opponents. Another area where analyzing NBA data has become important is in the betting industry. A report from 2018 indicated that the NBA could gain an additional $585 million dollars [thanks to legal betting](https://www.legalsportsreport.com/25173/aga-survey-mlb-nba-sports-betting/?doing_wp_cron=1670105815.3058269023895263671875). As legal betting expands worldwide, betting companies could employ machine learning models to offer a fair betting process. 

All that being said, it is suprisingly difficult to predict the outcome of NBA matches. Most models have similar performance at about 65-72%. Another interesting point is that the upset rate (a team with a better win-loss record losing to a team with a worse record) is of around 30%. Therefore, predicting the outcome of NBA matches still remains an interesting topics and many different approaches have been used to tackle this problem. 

# Dataset
Game data can be divided into regular and advanced statistics. Examples of regular statistics are points per game (PPG), two-point shooting percentage (2P%) and three-point shooting percentage (3P%). For advanced statistics, often a combination of different statistics is done. For example, true shooting percentage (TS%) takes into account two-, three- and free throw shooting. The purpose of using advanced statistics is to capture nuances about a team that cannot be seen using regular statistics - for example, taking into account the pace or the number of possessions a team has in a game.

For game outcome prediction, both regular and advanced team statistics were used. The data collected was from regular seasons games (playoffs not included) between the 2013 to 2021 seasons (over 10,000 matches). In a typical supervised machine learning problem, the outcome of the game was used as the label, and team statistics for both home and visitor team were used as features. 

For team win percentage prediction, only advanced team statistics was used. In this case, regular season data from 2001 to 2021 was used. The win percentage (number of wins over total number of games) in a given season was used as the target, and the team statistics for that season were used as features. 

# Hypotheses
Before actually working with the data, several hypotheses of what impacts game results were proposed: 
* Teams playing at home are more likely to win
* Higher efficiency when shooting (higher 2P%, 3P% or TS%) indicates teams that are more likely to win
* Teams that score a lot in a game are more likely to win (PPG or Offensive Rating - ORTG)
* Teams with strong defensive performances (Defensive Rating - DRTG) are more likely to win

One could consider the impact of these features in a machine learning model by looking at feature importance for a given model. It was also attempted to perform a qualitative analysis of these hypotheses during the exploratory data analysis (EDA) step. 

# Data Processing 
## Data Cleaning, EDA and Feature Engineering
This step involved exploring the game results and team statistics datasets, checking for missing values, evaluating features to use in the model and creating new features.

### Data Cleaning
Raw data had missing data in columns such as 'Notes', which indicated games that went to overtime or that were played abroad. This kind of information was not useful to the model, so it was dropped. Team names also had to be formatted to ensure the datasets could be merged. 

### EDA and Feature Engineering
An useful feature to create is the season. This is because when merging the game outcomes and team statistics, it is necessary to use the team statistics for the team in the season when the games occurred. 

