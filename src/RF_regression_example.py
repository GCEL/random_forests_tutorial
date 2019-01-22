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
# LOAD IN SOME DATA,
# Data Set Characteristics:
# Number of Instances: 506
# Number of Attributes: 13 numeric/categorical predictive. Median Value
#    (attribute 14) is usually the target.
# Attribute Information (in order):
# - CRIM per capita crime rate by town
# - ZN proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS proportion of non-retail business acres per town
# - CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - NOX nitric oxides concentration (parts per 10 million)
# - RM average number of rooms per dwelling
# - AGE proportion of owner-occupied units built prior to 1940
# - DIS weighted distances to five Boston employment centres
# - RAD index of accessibility to radial highways
# - TAX full-value property-tax rate per $10,000
# - PTRATIO pupil-teacher ratio by town
# - MEDV Median value of owner-occupied homes in $1000’s
# Creator:
#    Harrison, D. and Rubinfeld, D.L., 1980
#-------------------------------------------------------------------------------
data = load_boston()
X=data['data'][:,:11]
y=data['data'][:,-1]

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
plt.show()

#===============================================================================
# VARIABLE IMPORTANCES
#-------------------------------------------------------------------------------
# fit the full model
rf.fit(X,y)
cal_score_full = rf.score(X,y)
print("Calibration R$^2$ = %.02f" % cal_score_full)

# get variable importances
names = data["feature_names"][:11]
importances = rf.feature_importances_
print('Variables listed in order of importance')
print sorted(zip(map(lambda x: round(x, 4), importances), names), reverse=True)

#===============================================================================
# USING PARTIAL DEPENDENCIES TO UNDERSTAND FUNCTIONAL EFFECTS
#-------------------------------------------------------------------------------
# Importances tell you which variables were important in determining the target
# variable, but we still don't know how these explanatory variables effect the
# system (just that they are important). This is one drawback of using a non-
# parametric machine learning algorithm.
# One way to sort-of sidestep this difficulty is to use partial dependencies.
# Partial dependencies are the random forest equivalent to partial differentials
# of multivariate equations.
# We can plot partial dependencies by holding all variables but for the one of
# interest constant (e.g. the mean values for the other variables), then
# fitting the model across the range of the variable of interest. We'll try
# this now.
n_variables=X.shape[1]
RM_ = np.arange(np.min(X[:,5]),np.max(X[:,5])+1)
X_RM = np.zeros((RM_.size,n_variables))
for i in range(0,n_variables):
    if i == 5:
        X_RM[:,i] = RM_.copy()
    else:
        X_RM[:,i] = np.mean(X[:,i])

# predict with rf model
y_RM = rf.predict(X_RM)
# now plot
plt.figure(2, facecolor='White',figsize=[5,5])
ax = plt.subplot2grid((1,1),(0,0))
ax.plot(RM_, y_RM,'-')
ax.set_ylabel("Median house price / $1000s")
ax.set_xlabel("Average number of rooms")
plt.tight_layout()
plt.savefig('test_partial_deps_1.png')
plt.show()

# Now let's look at another variable - house age
AGE_ = np.linspace(np.min(X[:,6]),np.max(X[:,6]),100)
X_AGE = np.zeros((AGE_.size,n_variables))
for i in range(0,n_variables):
    if i == 6:
        X_AGE[:,i] = AGE_.copy()
    else:
        X_AGE[:,i] = np.mean(X[:,i])

# predict with rf model
y_AGE = rf.predict(X_AGE)
# now plot
plt.figure(3, facecolor='White',figsize=[5,5])
ax = plt.subplot2grid((1,1),(0,0))
ax.plot(AGE_, y_AGE,'-')
ax.set_xlabel("percentage homes built prior to 1940")
ax.set_ylabel("Median house price / $1000s")
plt.tight_layout()
plt.savefig('test_partial_deps_2.png')
plt.show()


# Really we would want to plot a number of lines for different combinations of
# the other explanatory variables, rather than just the average, since the
# relationship may not hold across the entire parameter space (and could be
# quite different!)
