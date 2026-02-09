# Milestone 1: Housing Price Model Building Project
# DSC 550-T303 Data Mining Assignment

# ------------------------------------------------------------------------------
# Business Problem Narrative (250–500 words)
# ------------------------------------------------------------------------------
# Title: Builder Incentive Optimization for New-Construction Pricing
#
# Business Problem and Context
# A regional homebuilder in Ames, Iowa, faces inconsistent demand for newly built
# homes across neighborhoods and seasons. Flat pricing strategies lead to
# overpricing some homes (higher carrying costs) and underpricing others (lost
# profit). The builder needs a data-driven pricing and incentive strategy that
# adapts to each home’s characteristics.
#
# Target Organization
# A mid-sized residential construction firm that builds 40–60 new homes per year
# in Ames. The firm decides how to price new builds at listing time and whether to
# offer incentives (upgraded finishes, garage expansion, or closing-cost credits)
# to accelerate sales without sacrificing margin.
#
# Problem Statement
# The builder requires a predictive model to estimate sale price for new builds
# and quantify the incremental value of builder-controlled features (e.g.,
# OverallQual, KitchenQual, GarageCars, TotalBsmtSF, GrLivArea). These insights
# allow the company to optimize pricing and select incentives that improve
# velocity and profitability.
#
# Model Target
# Target variable: SalePrice (continuous). Predictors: property attributes,
# quality ratings, square footage, garage capacity, and neighborhood.
#
# Expected Business Impact
# Feature-adjusted pricing can reduce time-on-market and improve gross margins.
# Quantifying upgrade value supports incentive packages with strong ROI. The final
# deliverable can be a pricing and incentive dashboard for rapid acquisition and
# pricing decisions.

# ------------------------------------------------------------------------------
# Import Packages
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# Load the Data
# ------------------------------------------------------------------------------
dir = '/Users/pyadav/Documents/DSC550-T303/Assignments/DataMining/house-prices-advanced-regression-techniques'
train_df = pd.read_csv(dir + '/train.csv')
test_df = pd.read_csv(dir + '/test.csv')

# ------------------------------------------------------------------------------
# Exploring the Data
# ------------------------------------------------------------------------------
train_df.head()
train_df.shape
train_df.dtypes
train_df.describe()
train_df.describe(include=['O'])

# ------------------------------------------------------------------------------
# Observations (add bullets, similar to Titanic notebook)
# ------------------------------------------------------------------------------
# • The dataset has 1460 rows and 81 columns (including SalePrice)
# • Target variable is SalePrice
# • Mix of numerical and categorical features
# • Missing values exist in several features

# ------------------------------------------------------------------------------
# Graphical Analysis (Minimum 4 graphs)
# ------------------------------------------------------------------------------
# Graph 1: SalePrice distribution
plt.rcParams['figure.figsize'] = (10, 5)
train_df['SalePrice'].hist(bins=40)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Count')
plt.show()

# Graph 2: SalePrice vs OverallQual
plt.rcParams['figure.figsize'] = (10, 5)
sns.boxplot(x='OverallQual', y='SalePrice', data=train_df)
plt.title('Sale Price by Overall Quality')
plt.show()

# Graph 3: SalePrice vs GrLivArea
plt.rcParams['figure.figsize'] = (10, 5)
plt.scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=0.5)
plt.title('Sale Price vs Above Grade Living Area')
plt.xlabel('GrLivArea (sq ft)')
plt.ylabel('Sale Price')
plt.show()

# Graph 4: Neighborhood median prices
plt.rcParams['figure.figsize'] = (12, 6)
neigh_median = train_df.groupby('Neighborhood')['SalePrice'].median().sort_values()
neigh_median.plot(kind='bar')
plt.title('Median Sale Price by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Median Sale Price')
plt.show()

# ------------------------------------------------------------------------------
# Graph Interpretation (write short analysis under each graph)
# ------------------------------------------------------------------------------
# Add narrative bullets like:
# • SalePrice is right-skewed; consider log transform later
# • Higher OverallQual strongly increases SalePrice
# • Larger living area correlates with higher SalePrice
# • Neighborhood is a major driver of price variability

# ------------------------------------------------------------------------------
# Conclusion (Short overview)
# ------------------------------------------------------------------------------
# Summarize key insights from graphs and how they inform the business problem.
