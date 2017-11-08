import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\yanlu\\Documents\\Data Science\\Week5-Visualization\\Week5-Visualization\\Indicators.csv')
print(data.shape)

print(data.head(10))
#4-d dataset country, indicator, year, value

#how many unique countries
countries = data['CountryName'].unique().tolist()
print(len(countries))

#are there same number of country codes
# How many unique country codes are there ? (should be the same #)
countryCodes = data['CountryCode'].unique().tolist()
print(len(countryCodes))
# How many unique indicators are there ? (should be the same #)
indicators = data['IndicatorName'].unique().tolist()
len(indicators)
# How many years of data do we have ?
years = data['Year'].unique().tolist()
len(years)
#range of years
print(min(years), "to", max(years))

##Plotting in matplotlib
# pick USA and indicator CO2 emissions
hist_indicator = 'CO2 emissions \(metric'
hist_country = 'USA'
mask1 = data['IndicatorName'].str.contains(hist_indicator)
mask2 = data['CountryCode'].str.contains(hist_country)
#stage is indicators matching the USA for country code and CO2 emissions
stage = data[mask1 & mask2]
print(stage.head())

#how emissions have changed
#get years
years = stage['Year'].values
#get values
co2 = stage['Value'].values
#create
plt.bar(years, co2)
plt.show()

#Improve graph
#a line plot
plt.plot(stage['Year'].values, stage['Value'].values)
#Label the axes
plt.xlabel('Year')
plt.ylabel(stage['IndicatorName'].iloc[0])
#Label the figure
plt.title('CO2 Emissions in the USA')
plt.axis([1959,2011,0,25])
plt.show()

##Using Histograms to explore the distribution of values
hist_data = stage['Value'].values
print(len(hist_data))
#histogram of the data
plt.hist(hist_data, 10, normed=False, facecolor='green')
plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Years')
plt.title('Histogram Example')
plt.grid(True) # add grid
plt.show()

#USA related to other countries
# select CO2 emissions for all countries in 2010
hist_indicator = 'CO2 emissions \(metric'
hist_year = 2010
mask1 = data['IndicatorName'].str.contains(hist_indicator)
mask2 = data['Year'].isin([hist_year])
# apply our mask
co2_2010 = data[mask1 & mask2]
co2_2010.head()
print(len(co2_2010))
#plot a histogram of the emissions per capita
# subplots returns a touple with the figure, axis attributes.
fig, ax = plt.subplots()
ax.annotate("USA",
            xy=(18, 5), xycoords='data',
            xytext=(18, 30), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
plt.hist(co2_2010['Value'], 10, normed=False, facecolor='green')

plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Countries')
plt.title('Histogram of CO2 Emissions Per Capita')

#plt.axis([10, 22, 0, 14])
plt.grid(True)
plt.show()