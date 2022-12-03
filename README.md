# Predicting the outcome of NBA games
Final project as part of the Lighthouse Labs Data Science Bootcamp. Data used in this project was obtained from [Basketball Reference](https://www.basketball-reference.com/).

## Introduction 
The NBA is the main basketball league around the world. The league consists of 30 teams divided into two conferences, and a team plays every other team at least twice. Overall, each team plays a total of 82 games over the course of the regular season. 

A lot of data can be generated during NBA games, for example player and team statistics. This data can be used in several different areas, from player performance analysis to devising a team's strategy against specific opponents. Another area where analyzing NBA stats has become important is in the betting industry. With the worldwide expansion of legal betting, betting companies could employ machine learning models to offer a fair betting process. 

In general, NBA statistics can be divided into regular and advanced statistics. Examples of regular statistics are points per game (PPG), two-point shooting percentage (2P%) and three-point shooting percentage (3P%). For advanced statistics, often a combination of different statistics is done. For example, true shooting percentage takes into account two-, three- and free throw shooting. The purpose of using advanced statistics is to capture nuances about a team that cannot be captured using regular statistics - for example, taking into account the pace or the number of possessions a team has in a game.

All that being said, it is suprisingly difficult to predict the outcome of NBA matches. Most models have similar performance at about 65-72%. Another interesting point is that the upset rate (a team with a better win-loss record losing to a team with a worse record) is of around 30%. 

In this project, both regular and advanced team statistics are used in other to create machine learning model capable of predicting the outcome of NBA games. This is a popular topic that has been approached in several different ways. 