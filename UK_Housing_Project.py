#!/usr/bin/env python
# coding: utf-8

# # Project Introduction
# 
# In this project, I'm going to analyze UK housing prices from 1978 to 2023 using the Local Authority Housing Statistics (LAHS) dataset. The goal is to identify trends, patterns, and factors that influence price changes over time. By understanding these patterns, I aim to provide insights that can assist policymakers, urban planners, and homebuyers in making informed decisions. I will clean and preprocess the data to ensure accuracy, explore trends through visualizations, and apply a regression model to investigate the factors that affect housing prices.
# 
# ---
# 
# Let's start by importing the libraries we will use in this project.
# 

# In[55]:


# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm



# Load the dataset

# In[56]:


df = pd.read_csv("/content/LAHS_open_data_1978-79_to_2022-23.csv")


# Display the first few rows of the dataset to understand its structure

# In[57]:


df.head()


# ---
# 
# As we can see from the results, the dataset contains a variety of columns, each representing different aspects of housing data, such as region names, local authority codes, and various numerical indicators.
# 
# ---
# 
# Next, let's display a summary of the dataset to see the data types and check for any potential issues in the column types.
# 
# 

# In[58]:


df.info()


# 
# ---
# 
# Our dataset contains a total of 15,483 records and 35 columns, with several fields having missing values that we will handle in the data cleaning step.
# 
# ---
# 
# 
# Let's check for missing values to identify columns that may need cleaning
# 
# 

# In[59]:


missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)


# ---
# 
# From the missing values summary, we see that `metropolitan_county_name` and `metropolitan_county_code` have a significant amount of missing data, while other columns are mostly complete.
# 
# ---
# 
# We saw earlier that some columns contain `[z]` and `[x]`, which represent missing values. Let's replace these placeholders with `NaN` to standardize missing values across the dataset.
# 
# 

# In[60]:


# Replace '[z]' and '[x]' with NaN
df.replace({'[z]': pd.NA, '[x]': pd.NA}, inplace=True)


# Let's drop specified columns with too many missing values and columns that have 0 as values in all rows

# In[61]:


df.drop(columns=["h9a", "f1a", "i2ca"], inplace=True)


# Now we define the columns to convert to numeric and fill NaN values with the mean

# In[62]:


columns_to_fill = ["a2ia", "a2iaa", "a2iab", "a4c", "a4d", "d4a", "d8a", "i3fa", "i3fb", "h5a",
                   "h8a", "e1a", "f16a", "f16b", "i13a", "i14a", "j1aa", "j1ab", "i1a", "i2a"]

# Convert only the specified columns to numeric, coercing errors to NaN
df[columns_to_fill] = df[columns_to_fill].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the mean for the specified columns
df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].mean())


# After going through the dataset documentation, I decided to rename the alphanumeric identifiers with their respective descriptive names based on the documentation notes.

# In[63]:


# Apply column renaming using the provided column mapping dictionary
column_mapping = {
    "a2ia": "Total_All_Dwellings",
    "a2iaa": "Total_Social_Rent_Dwellings",
    "a2iab": "Total_Affordable_Rent_Dwellings",
    "a4c": "New_Builds",
    "a4d": "Acquisitions",
    "d4a": "Total_Lettings_Existing_Social_Tenants",
    "d8a": "Total_Dwellings_Let_New_Tenants_Social_Housing",
    "i3fa": "LA_Total_Units_Without_Developer_Contributions",
    "i3fb": "LA_Total_Units_With_Developer_Contributions",
    "h5a": "Current_Tenants_Arrears",
    "h8a": "Rent_Arrears_Written_Off",
    "e1a": "Vacant_Dwellings_LA_Owned",
    "f16a": "Total_Non_Decent_Dwellings",
    "f16b": "Total_Cost_To_Make_Dwellings_Decent",
    "i13a": "Financial_Contributions_s106_Received",
    "i14a": "Financial_Contributions_s106_Spent",
    "j1aa": "LA_New_Social_Rent_Start_Without_Developer_Contributions",
    "j1ab": "LA_New_Social_Rent_Start_With_Developer_Contributions",
    "i1a": "Units_Completed_Populations_Less_Than_3000",
    "i2a": "Units_On_Rural_Exception_Sites"
}
df.rename(columns=column_mapping, inplace=True)


# To ensure consistency, I will parse the 'Year' column by extracting the starting year from each range and converting it to a datetime format.

# In[64]:


# Ensure dates are parsed correctly by extracting the starting year from 'Year'
df['Year'] = df['Year'].str.split('-').str[0]  # Extract the first part of the range
df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')  # Convert to datetime format


# In[65]:


df.head()


# Finally, we will save the processed data to avoid redundant cleaning steps in future analyses.

# In[66]:


# Save the processed data to avoid redundant cleaning in future steps
df.to_csv("processed_data.csv", index=False)


# Now we have a clean dataset ready for analysis. Data preparation and cleaning were completed successfully.
# 
# ---
# 
# ### Exploratory Data Analysis (EDA)
# 
# In this section, I will explore key characteristics of our dataset. My objective is to gain initial insights and identify any trends or patterns that may be relevant to the study of housing prices. This will involve calculating descriptive statistics, exploring correlations, and visualizing data distributions and outliers.
# 

# **Summary Statistics**
# 
# To start, let's examine the summary statistics for key columns in our dataset. This will help us understand the range, central tendencies, and variability of our data.

# In[67]:


# Load the processed dataset
df = pd.read_csv("/content/processed_data.csv")

# Display summary statistics for key columns to understand central tendencies and range
key_columns = [
    "Total_All_Dwellings", "Total_Social_Rent_Dwellings", "Total_Affordable_Rent_Dwellings",
    "New_Builds", "Acquisitions", "Total_Lettings_Existing_Social_Tenants",
    "Total_Dwellings_Let_New_Tenants_Social_Housing", "LA_Total_Units_Without_Developer_Contributions",
    "LA_Total_Units_With_Developer_Contributions", "Current_Tenants_Arrears", "Rent_Arrears_Written_Off",
    "Vacant_Dwellings_LA_Owned", "Total_Non_Decent_Dwellings", "Total_Cost_To_Make_Dwellings_Decent",
    "Financial_Contributions_s106_Received", "Financial_Contributions_s106_Spent",
    "LA_New_Social_Rent_Start_Without_Developer_Contributions", "LA_New_Social_Rent_Start_With_Developer_Contributions",
    "Units_Completed_Populations_Less_Than_3000", "Units_On_Rural_Exception_Sites"
]

# Display summary statistics
df[key_columns].describe()


# The summary statistics reveal that several columns, such as Total_Affordable_Rent_Dwellings, New_Builds, and Total_Dwellings_Let_New_Tenants_Social_Housing, exhibit high variability and extreme values. This may indicate disparities in housing availability, conditions, and support across different local authorities, which warrants further exploration in our analysis.
# 
# 
# 
# 
# 
# 
# 

# **Correlation Analysis**
# 
# Next, let's examine the correlation matrix for these key columns. This matrix will help us understand the relationships between different variables, shedding light on potential influences on housing attributes.

# In[68]:


# Compute correlation matrix for the selected columns
correlation_matrix = df[key_columns].corr()
correlation_matrix


# From the correlation matrix, we can observe some interesting relationships among the variables. For instance, there is a moderate positive correlation between Total_Affordable_Rent_Dwellings and Acquisitions (0.42), indicating that areas with more affordable rent dwellings tend to have higher acquisitions. Additionally, Total_Lettings_Existing_Social_Tenants and Total_Dwellings_Let_New_Tenants_Social_Housing have a strong positive correlation (0.56), suggesting that these variables often increase together, likely reflecting similar social housing trends. There are also some weaker correlations with minimal influence, which will likely have less impact in the further analysis.

# In[69]:


# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Key Housing Attributes")
plt.show()


# From the heatmap, we can get a clear view of correlations among different housing attributes. High positive or negative correlations can suggest relationships between attributes that may be worth investigating further.

# **Visualizations**
# 
# To see the data distribution and identify any outliers, lets plot histograms for the total dwellings, social rent dwellings, and affordable rent dwellings, followed by box plots for these variables to assess their variability and spot potential outliers.

# In[70]:


# Step 3: Visualizations

# Histogram for distribution analysis of total dwellings, social rent dwellings, and affordable rent dwellings
plt.figure(figsize=(15, 5))
for i, column in enumerate(["Total_All_Dwellings", "Total_Social_Rent_Dwellings", "Total_Affordable_Rent_Dwellings"], 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f"Distribution of {column.replace('_', ' ')}")
plt.tight_layout()
plt.show()


# From the histograms, we can see that most values for Total_All_Dwellings, Total_Social_Rent_Dwellings, and Total_Affordable_Rent_Dwellings are concentrated at the lower end, with very few entries at the higher values. This distribution indicates a high frequency of lower dwelling counts across regions, suggesting that large numbers of dwellings are less common.

# In[71]:


# Box Plots to identify outliers in price-related columns
plt.figure(figsize=(15, 5))
for i, column in enumerate(["Total_All_Dwellings", "Total_Social_Rent_Dwellings", "Total_Affordable_Rent_Dwellings"], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df[column])
    plt.title(f"Box Plot of {column.replace('_', ' ')}")
plt.tight_layout()
plt.show()


# The box plots reveal that Total_All_Dwellings, Total_Social_Rent_Dwellings, and Total_Affordable_Rent_Dwellings contain numerous outliers, indicating that most values are low with a few significantly higher entries in each category.

# **Analyze the Average Housing Prices Across Neighborhoods**
# 
# **Goal:** To identify price differences between neighborhoods.

# **Calculate Averages**
# 
# First, let's group the data by region and calculate the average housing prices for the year 2021 to observe any regional price disparities.

# In[72]:


# Calculate Averages: Let's group data by region and calculate the average housing prices for the year 2021
year = '2021'
filtered_data = df[df['Year'].str.contains(year)]

# Calculate the average of 'Total_All_Dwellings' by region
avg_housing_prices = filtered_data.groupby('region_name')['Total_All_Dwellings'].mean().sort_values(ascending=False)

# Display the average housing prices for verification
print("Average Housing Prices by Region for the Year", year)
print(avg_housing_prices)


# The North East stands out with a significantly higher average housing price than all other regions, which are clustered around similar lower values. This suggests that the North East might have unique factors, such as property demand or availability of social housing, that influence its higher average price.

# To visually represent the regional differences, we will plot a bar chart of the average housing prices by region for the year 2021.

# In[73]:


# Use a bar chart to display average housing prices by neighborhood/region for the selected year
plt.figure(figsize=(10, 6))
avg_housing_prices.plot(kind='bar', color='skyblue')
plt.title(f"Average Housing Prices by Region in {year}")
plt.xlabel("Region")
plt.ylabel("Average Housing Price (Total All Dwellings)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# As seen in the bar chart, the North East region shows a a huge difference in average housing prices compared to other regions, suppoting the numerical findings.

# **Visualize Trends Over Time**
# 
# To understand trends over time, we plot the average housing prices across all regions from the start year to the latest available year.

# In[74]:


# Plot the trends over time for each region
df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year

# Calculate average housing prices by region and year
avg_prices_by_region_year = df.groupby(['Year', 'region_name'])['Total_All_Dwellings'].mean().unstack()

plt.figure(figsize=(14, 8))
for region in avg_prices_by_region_year.columns:
    plt.plot(avg_prices_by_region_year.index, avg_prices_by_region_year[region], label=region)

plt.title("Average Housing Prices by Region Over Time")
plt.xlabel("Year")
plt.ylabel("Average Housing Price (Total All Dwellings)")
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# The line plot shows that the North East region experienced a substantial increase in average housing prices over time, especially in recent years. Other regions have relatively stable trends with minimal fluctuations, indicating regional factors driving prices in the North East specifically.
# 
# 
# 
# 
# 
# 
# 

# **Study the Impact of Government Housing Policies on Regional Housing Prices**
# 
# In this section, let's take a look at how government policies impact regional housing prices. By comparing housing price trends in regions with high versus low government intervention, we can assess whether policies significantly influence affordability.
# 

# We start by selecting the policy-related variables that represent government intervention in housing, such as financial contributions and social rent starts. We also include `Total_All_Dwellings` to track housing prices, along with `region_name` and `Year` to allow for regional analysis.
# 

# In[ ]:


# Define policy-related variables
policy_features = [
    'Financial_Contributions_s106_Received',
    'Financial_Contributions_s106_Spent',
    'LA_New_Social_Rent_Start_Without_Developer_Contributions',
    'LA_New_Social_Rent_Start_With_Developer_Contributions'
]

# Add 'Total_All_Dwellings' to capture housing prices
features = policy_features + ['Total_All_Dwellings', 'region_name', 'Year']


# Here, we create a subset of the dataset that includes only the selected features. This subset will serve as the foundation for our analysis, focusing specifically on policy-related variables and housing prices.
# 

# In[ ]:


# Select relevant columns for the analysis and create a copy
policy_df = df[features].copy()


# To quantify the level of government intervention, we calculate a `Total_Policy_Intervention` score for each region and year by summing up the values of the selected policy-related variables.
# 

# In[ ]:


# Calculate total government intervention as the sum of all policy-related variables
policy_df['Total_Policy_Intervention'] = policy_df[policy_features].sum(axis=1)


# To compare regions with different levels of intervention, we categorize them into "High Intervention" and "Low Intervention" groups based on the median intervention score. This classification allows us to observe potential differences in housing price trends between these two groups.
# 

# In[ ]:


# Calculate the median intervention level
median_intervention = policy_df['Total_Policy_Intervention'].median()

# Create a new column to categorize regions based on policy intervention
policy_df['Intervention_Level'] = policy_df['Total_Policy_Intervention'].apply(
    lambda x: 'High Intervention' if x >= median_intervention else 'Low Intervention'
)
print(policy_df['Intervention_Level'].value_counts())


# The data shows that regions classified as "High Intervention" outnumber those in "Low Intervention" by nearly two-to-one, indicating more areas received substantial policy support.
# 

# We aggregate the data by year and intervention level to calculate the average housing prices for each group. This provides a clearer view of how housing prices vary over time based on the level of government intervention.
# 

# In[ ]:


# Group by year, intervention level, and calculate the mean housing prices
policy_trends = policy_df.groupby(['Year', 'Intervention_Level'])['Total_All_Dwellings'].mean().reset_index()
print(policy_trends.head(-5))


# From the yearly average housing prices table, regions with high government intervention maintained relatively stable prices over time, with notable increases only after 2015.
# 

# Plot Housing Prices Over Time

# In[ ]:


# Plot the data
plt.figure(figsize=(12, 6))
for label, df_sub in policy_trends.groupby('Intervention_Level'):
    plt.plot(df_sub['Year'], df_sub['Total_All_Dwellings'], label=label)

# Customize the plot
plt.title('Housing Prices Over Time: High vs. Low Government Policy Intervention')
plt.xlabel('Year')
plt.ylabel('Average Housing Prices (Total_All_Dwellings)')
plt.legend(title='Intervention Level')
plt.grid(True)
plt.show()


# The plot reveals that since 2010, regions with high government intervention tend to have higher housing prices compared to those with low intervention, suggesting a potential correlation between policy support and increased housing prices over time.
# 

# **Analyze Urban Trends and Their Correlation with House Prices in Major Cities**
# 
# In this section, I’ll analyze urban trends and their correlation with house prices in major cities. By categorizing local authorities into urban and rural classifications, I aim to uncover any significant price differences and trends over time.

# Let's start by defining which authorities we consider urban based on the datset documentation. We will then categorize all authorities as either urban or rural.

# In[ ]:


# Define a simple list to classify certain authorities as urban based on general knowledge
# In practice, we would ideally have population size data for more accurate classification
urban_authorities = [
    'Birmingham', 'Manchester', 'Liverpool', 'Leeds', 'Sheffield', 'Bristol',
    'Nottingham', 'Leicester', 'Coventry', 'Bradford', 'Wakefield', 'Cardiff',
    'Sunderland', 'Newcastle upon Tyne', 'Kingston upon Hull', 'Southampton',
    'Portsmouth', 'London'
]

# Create a new column 'Urban_Rural' in the DataFrame based on the classification
df['Urban_Rural'] = df['local_authority'].apply(lambda x: 'Urban' if x in urban_authorities else 'Rural')

# Display the count of Urban and Rural classifications for validation
print(df['Urban_Rural'].value_counts())


# As we can see, the majority of areas are classified as Rural, while a smaller portion falls under Urban.

# Next, we'll group the data by year and urban/rural classification to calculate the average housing prices over time for each category.

# In[ ]:


# Group data by Year and Urban/Rural classification to get the average housing prices over time
urban_rural_price_trends = df.groupby(['Year', 'Urban_Rural'])['Total_All_Dwellings'].mean().reset_index()

# Display the first few rows to confirm the grouped data
print(urban_rural_price_trends.head(-5))


# We now have average housing prices segmented by urban and rural classifications over time, which helps us observe trends in each area type.

# Let's visualize the average housing prices over time for urban and rural areas to see if any notable trends appear.

# In[ ]:


# Plotting average housing prices over time for Urban and Rural areas
plt.figure(figsize=(12, 6))
for label, df_sub in urban_rural_price_trends.groupby('Urban_Rural'):
    plt.plot(df_sub['Year'], df_sub['Total_All_Dwellings'], label=label)

# Customize the plot
plt.title('Housing Prices Over Time: Urban vs. Rural Areas')
plt.xlabel('Year')
plt.ylabel('Average Housing Prices (Total_All_Dwellings)')
plt.legend(title='Area Type')
plt.grid(True)
plt.show()


# From the plot above, we can observe fluctuations in housing prices in urban and rural areas, with urban areas showing sharper changes in recent years.

# Finally, let's examine the relationship between the number of new builds and housing prices, using a scatter plot to highlight any patterns between urban and rural areas.

# In[ ]:


# Scatter plot showing the relationship between new builds and housing prices
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='New_Builds', y='Total_All_Dwellings', hue='Urban_Rural', alpha=0.6)
plt.title("Relationship Between New Builds and Housing Prices")
plt.xlabel("New Builds")
plt.ylabel("Housing Prices (Total_All_Dwellings)")
plt.legend(title='Area Type')
plt.show()



# The scatter plot suggests that there’s no clear linear relationship between the number of new builds and housing prices, but urban areas appear to have slightly higher prices overall.

# **Identify Factors Influencing Price Changes**
# 
# In this section, I am going to identify factors influencing housing price changes by conducting multiple linear regression. This analysis will help us quantify the impact of various features on housing prices.
# 
# 
# Let's start by defining the Dependent and Independent Variables.
# 
# we define Total_All_Dwellings as the dependent variable representing housing prices. The independent variables include a selection of features that might influence housing prices, such as new builds, acquisitions, and affordable rent dwellings.

# In[75]:


# Define the dependent and independent variables
target = 'Total_All_Dwellings'  # Dependent variable representing housing prices

features = [
    "Total_Social_Rent_Dwellings", "Total_Affordable_Rent_Dwellings",
    "New_Builds", "Acquisitions", "Total_Lettings_Existing_Social_Tenants",
    "Total_Dwellings_Let_New_Tenants_Social_Housing", "LA_Total_Units_Without_Developer_Contributions",
    "LA_Total_Units_With_Developer_Contributions", "Current_Tenants_Arrears", "Rent_Arrears_Written_Off",
    "Vacant_Dwellings_LA_Owned", "Total_Non_Decent_Dwellings", "Total_Cost_To_Make_Dwellings_Decent",
    "Financial_Contributions_s106_Received", "Financial_Contributions_s106_Spent",
    "LA_New_Social_Rent_Start_Without_Developer_Contributions", "LA_New_Social_Rent_Start_With_Developer_Contributions",
    "Units_Completed_Populations_Less_Than_3000", "Units_On_Rural_Exception_Sites"
]

# Prepare the data
X = df[features]
y = df[target]


# We then split the data into training and testing sets for analysis.
# 
# 

# In[76]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# We then define the MultipleLinearRegressionStatsModel helper class to enhance the model’s accuracy by using stepwise feature selection, allowing us to analyze the relationship between features and housing prices more clearly than with the standard sklearn class.

# In[77]:


# Define the helper class with feature selection
class MultipleLinearRegressionStatsModel:
    def __init__(self, feature_names, target_name, feature_selection=False):
        self.feature_names = feature_names
        self.target_name = target_name
        self.model = None
        self.results = None
        self.feature_selection = feature_selection
        self.selected_features = feature_names

    def fit(self, X, y):
        # Feature selection with stepwise method
        if self.feature_selection:
            sfs = SequentialFeatureSelector(estimator=LinearRegression(), forward=True, k_features='best', scoring='r2', cv=5, n_jobs=-1)
            sfs.fit(X, y)
            self.selected_features = list(sfs.k_feature_names_)
        else:
            self.selected_features = self.feature_names

        # Fit the linear regression model using statsmodels
        X = sm.add_constant(X[self.selected_features])
        self.model = sm.OLS(y, X)
        self.results = self.model.fit()

    def predict(self, X_data):
        # Add constant term to new data and make predictions
        X_data = sm.add_constant(X_data[self.selected_features])
        return self.results.predict(X_data)

    def get_summary(self):
        summary_data = {
            'Dep. Variable': self.target_name,
            'No. Observations': len(self.results.model.endog),
            'Model': 'OLS',
            'Df Residuals': self.results.df_resid,
            'R-squared': self.results.rsquared,
            'Adj. R-squared': self.results.rsquared_adj,
            'F-statistic': self.results.fvalue,
            'Prob (F-statistic)': self.results.f_pvalue,
            'Log-Likelihood': self.results.llf,
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'Standard Error of Estimate': self.results.mse_resid**0.5
        }
        summary = pd.DataFrame([summary_data])
        return summary

    def get_coef_summary(self):
        coef_summary_data = {
            'Coefficient Value': self.results.params,
            'Standard Error': self.results.bse,
            't-value': self.results.tvalues,
            'P>|t|': np.round(self.results.pvalues,4)
        }
        coef_summary = pd.DataFrame(coef_summary_data)
        return coef_summary


# In this class, we use SequentialFeatureSelector to perform stepwise selection to identify the most relevant features for predicting housing prices. The class also provides methods to retrieve the regression summary and coefficient details for interpretation.

# Fit the Model and Display the Summary and Coefficient Summary

# In[78]:


# Create an instance of the regression model with feature selection enabled
reg_model = MultipleLinearRegressionStatsModel(feature_names=features, target_name=target, feature_selection=True)

# Fit the model using the training data
reg_model.fit(X_train, y_train)

# Get the model summary and coefficient summary
model_summary = reg_model.get_summary()
coef_summary = reg_model.get_coef_summary()

# Print the model summary
print("Multiple Linear Regression Summary:")
print(model_summary.transpose())

# Print the coefficient summary
print("\nRegression Coefficients Summary:")
coef_summary


# ## Results Analysis:
# 
# From the multiple linear regression model results, I got insights into the factors influencing housing prices, as represented by the `Total_All_Dwellings` variable. Below is an interpretation of the key metrics and coefficients:
# 
# ### Model Summary:
# - **R-squared (0.1633)**: This indicates that about 16.33% of the variance in `Total_All_Dwellings` is explained by the selected features. Although this value is relatively low, it suggests that there may be other unaccounted factors influencing housing prices. However, in complex social and economic data, lower R-squared values are common.
# - **F-statistic (226.44, p-value = 0.0)**: The F-statistic and its corresponding p-value (0.0) suggest that the model as a whole is statistically significant, indicating that the predictors are jointly explaining a non-zero portion of the variance in the dependent variable.
# - **AIC and BIC**: The Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) provide measures for model selection, with lower values indicating a better fit when comparing models. Here, they help in evaluating the overall model quality.
# 
# ### Coefficient Analysis:
# - **Intercept (0.612)**: The baseline value when all predictor variables are zero. It provides a reference point but has limited interpretability on its own.
# - **Total_Social_Rent_Dwellings (1.022, p < 0.0001)**: This feature has a positive and significant effect on housing prices. For each additional unit in social rent dwellings, the average housing price increases by 1.022 units, holding other factors constant. This indicates a strong positive relationship between social rent dwellings and overall housing prices.
# - **LA_Total_Units_With_Developer_Contributions (0.080, p < 0.0001)**: This variable is also statistically significant, suggesting that units built with developer contributions are associated with higher housing prices, potentially due to enhanced infrastructure and amenities.
# - **Current_Tenants_Arrears (11.522, p < 0.0001)**: This coefficient indicates a significant positive relationship, meaning areas with higher tenant arrears are correlated with higher housing prices. This may reflect socioeconomic factors that influence both arrears and price.
# 
# #### Other Factors:
# - Variables like `Total_Lettings_Existing_Social_Tenants`, `LA_Total_Units_Without_Developer_Contributions`, `Total_Cost_To_Make_Dwellings_Decent`, and others show low or statistically insignificant coefficients. Their high p-values suggest that they may not have a strong influence on the dependent variable, at least within this model's context.
# - **Negative Coefficients**: Some factors, such as `Total_Cost_To_Make_Dwellings_Decent`, exhibit negative coefficients, indicating a slight inverse relationship. However, the statistical insignificance of these variables (p > 0.05) means we cannot confidently conclude their impact on housing prices.
# 
# ### Conclusion:
# - Key factors influencing housing prices include `Total_Social_Rent_Dwellings`, `LA_Total_Units_With_Developer_Contributions`, and `Current_Tenants_Arrears`. These factors have significant coefficients and relatively low p-values, indicating a meaningful impact on housing prices.
# - The low R-squared value suggests that while these factors provide some insight, additional features or alternative models might better capture the complexity of housing prices.
# 

# Let's now evaluate Performance on Test Data

# In[79]:


# Use predict() to make predictions on the test set with selected features
y_pred = reg_model.predict(X_test)

# Display performance metrics on the test data
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nPerformance Metrics on Test Data:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")


# The test data shows an MSE of 1331.68, RMSE of 36.49, and MAE of 0.95, indicating moderate accuracy and potential for improvement.
# 

# Let's visualize Residuals to Check Distribution

# In[80]:


all_residuals = y_test - y_pred
mean, std = np.mean(all_residuals), np.std(all_residuals)

fig, ax = plt.subplots()
ax.hist(all_residuals, bins=25, edgecolor='black', alpha=0.7, density=True)

# Customize labels and title
ax.set_title('Residuals')
ax.set_xlabel('Values')
ax.set_ylabel('Frequency')

# Plot normal distribution overlay
x = np.linspace(mean - 3*std, mean + 3*std, 100)
pdf = norm.pdf(x, mean, std)
ax.plot(x, pdf, 'k', linewidth=2, label='Normal distribution')

plt.legend()
plt.show()


# The residuals plot shows a distribution centered around zero, indicating a reasonable model fit with some deviations from normality.
# 
