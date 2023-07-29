# COMP9417-23T2-project
UNSW COMP9417 23T2 GROUP PROJECT
### 1. Download data from Kaggle and save it in input directory
https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data  
### 2. Load data
Run `load_input_data.py`, it will save `dataset_df.pkl`, `labels.pkl` and `true.pkl` in the data directory.
### 3. Defining Feature Engineering
The `feature_engineer.py` defines two feature engineering functions to be used for training the model.
### 4. Training Models
In the baselines directory, four models are defined, each applying two different feature engineering functions. Each IPynb saves their models and predictions in the data directory, e.g. `GBT_FE1_models.pkl`, `GBT_FE1_predictions.csv`.
### 5. Stacking models
Eventually, we stacked the four best-performing models and arrived at the final f1 scores, the results are saved in `stacking_FE2.ipynb` file.
### References
[numpy](https://numpy.org/)  
[pandas](https://pandas.pydata.org/)  
[pickle](https://docs.python.org/3/library/pickle.html)  
[matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)  
[SciPy - minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)  
[scikit-learn - f1_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)  
[scikit-learn - GroupKFold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)  
[scikit-learn - LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
[scikit-learn - MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)  
[TensorFlow - Decision Forests](https://www.tensorflow.org/decision_forests)  
[TensorFlow - Keras](https://keras.io/)    
[LightGBM - LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)  
[XGBoost - XGBClassifier](https://xgboost.readthedocs.io/)  
