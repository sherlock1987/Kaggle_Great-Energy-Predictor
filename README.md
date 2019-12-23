# DATA:
wget https://drive.google.com/open?id=1m0VSwFo1OVPbivmfhwvwRaw7FaDQd7U4


# How to run
Simply run the code main.py, and the code will load the model and data to test. If you want to check the score 
in Kaggle, simply upload the output.csv, and you will see the results.


# CODE DESCRIPTION

|---main.py Reproduce the test  results.

	|---Adaboost.py Training and validating process for adaboost algorithm
	|---DTR_Model.py Training and validating process for decision tree algorithm
	|---prophet_validation.py Training and validating process for prophet algorithm
	|---LGBM.py Training and validating process for Lightgbm algorithm
	
|---model where model saved, basically three model, but work together to predict.

|---feature_engineering

	|---building_procesing.py To solve the missing data in building dataset
	|---weather_preprocessing.py To solve the missing data in weather dataset
	|---merge.py Merge building data and weather data
	|---draw_graph Draw the plot which need in the report
	|---sample_maker make a sample dataset for use
	
|---experiment_details

	|---Adaboost.pdf The experiment details for adaboost
	|---DTR.pdf The experiment details for Decision Tree
	|---prophet The experiment results for Prophet model
	|---LGBM.png  The experiment results for LGBM model

