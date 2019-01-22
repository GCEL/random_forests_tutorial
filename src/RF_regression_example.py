"""
21/01/2019 - DTM
Example random forest regression
"""

# Import some python libraries
import numpy as np
import pandas as pd

# Import relavent parts of scikit-learn library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

# Import some plotting libraries
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

#===============================================================================
# LOAD IN SOME DATA, saved as a pandas dataframe
# - AGB (Mg/ha)
# - Latitiude
# - Longitude
# - WorldClim climatology estimates (10 variables)
# - Soilgrids soil texture estimates (3 variables)
# - flag: training data = 1
#-------------------------------------------------------------------------------
data = np.load('filename.npz') # load data
data = load_boston()
X=data['data'][:,:13]
y=data['data'][:,-1]
#===============================================================================
# PREPROCESSING
#-------------------------------------------------------------------------------
training_mask = data[:,-1] == 1  # a mask to find rows that are marked as training data
X = data[training_mask,1:-1] # this takes all training data rows, and the data from the third column to the penulitimate column
y = data[training_mask,0]

#===============================================================================
# CAL-VAL
#-------------------------------------------------------------------------------
# Split the training data into a calibration and validation set using the scikit learn tool
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5)

#create the random forest object with predefined parameters
# *** = parameters that often come out as being particularly sensitive
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth=None,            # ***maximum number of branching levels within each tree
            max_features='auto',       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=0.0, # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=5,#20,,       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=2,       # ***The minimum number of samples required to split an internal node
            min_weight_fraction_leaf=0.0,
            n_estimators=100,          # ***Number of trees in the random forest
            n_jobs=-1,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=None,         # seed used by the random number generator
            verbose=0,
            warm_start=False)

# fit the calibration sample
rf.fit(X_train,y_train)
y_train_rf = rf.predict(X_train)
cal_score = rf.score(X_train,y_train) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R$^2$ = %.02f" % cal_score)

# fit the validation sample
y_test_rf = rf.predict(X_test)
val_score = rf.score(X_test,y_test)
print("Validation R$^2$ = %.02f" % val_score)

# Plot the calibration and validation data
# - First put observations and model values into dataframe for easy plotting with seaborn functions
calval_df = pd.DataFrame(data = {'val_obs': y_test,
                                 'val_model': y_test_rf,
                                 'cal_obs': y_train,
                                 'cal_model': y_train_rf})

plt.figure(1, facecolor='White',figsize=[8,4])
ax_a = plt.subplot2grid((1,2),(0,0))
sns.regplot(x='cal_obs',y='cal_model',data=calval_df,marker='+',
            truncate=True,ci=None,ax=ax_a)
ax_a.annotate('calibration R$^2$ = %.02f\nRMSE = %.02f' %
            (cal_score,np.sqrt(mean_squared_error(y_train,y_train_rf))),
            xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='left', verticalalignment='top')
ax_b = plt.subplot2grid((1,2),(0,1),sharex = ax_a)
sns.regplot(x='val_obs',y='val_model',data=calval_df,marker='+',
            truncate=True,ci=None,ax=ax_b)
ax_b.annotate('validation R$^2$ = %.02f\nRMSE = %.02f'
            % (val_score,np.sqrt(mean_squared_error(y_test,y_test_rf))),
            xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='left', verticalalignment='top')
ax_a.axis('equal')
plt.savefig('test_rf.png')


# fit the full model
