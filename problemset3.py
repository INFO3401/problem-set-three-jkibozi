# INFO 3401
# Problem Set 3
# Jason Kibozi-Yocka

import pandas as pd
import matplotlib.pyplot as plt

# 1. Download the data in the Data/Financial Data folder on Canvas. Open the file and take a peek at the data dictionary. What do you think this data is used for? 

print("Q1 Answer: This is used to assess the likelihood that a borrower will pay back their loans (aka loan security or risk of investment).")

# 2. Create a function called loadAndCleanData that takes as an argument a filename and returns a Pandas dataframe. That dataframe should contain the data from the CSV file cleaned such that any cells missing data, containing a NaN value or the string "NA" are filled with 0s (this is a technique called zero-filling that we will talk about shortly!)

def loadAndCleanData(myFile):
	myDF = pd.read_csv(myFile)
	myDF.fillna(value=0,inplace=True)
	return myDF

# 3. Add a line to your Python file that uses the function to load in the creditData.csv file from Canvas when the Python script is run.

myDataframe = loadAndCleanData("creditData.csv")

# 4. Now that you've got your data loading, you can generate probability density functions for each feature. These PDFs will tell you the probability of a given feature occurring based on our data. You can use Kernel Density Estimation (KDE) to do this. Write a function called computePDF that takes as arguments a target feature and a dataset and generates a KDE plot for each feature in your data (hint: check out the plot.kde function here (Links to an external site.)). You will need to import matplotlib.pyplot as plt and use plt.show() to make the graphs appear. Call that feature on each column of your dataset when you run your script.

def computePDF(myFeature,myDataset):
	myPlot = myDataset[myFeature].plot.kde()
	plt.show()
'''
for feature in myDataframe.columns.values:
	computePDF(feature,myDataframe)
'''
# 5. Given the skews that you see in your data, you might want to step back and take a look at what's actually in your data. You can look at the distribution of values in the columns. This will help you understand what data you have. To do this, write a function called viewDistribution that takes in the name of a column and a dataframe and shows a histogram of values in that column (hint: check out the hist function here (Links to an external site.)).  Comment out your computePDF function call and instead use viewDistribution to look at the distribution of each column in your dataset when the Python script is run. This should come after you call the loadAndCleanData function. Notice anything strange about some of these histograms?

def viewDistribution(myCol,myDataset):
	myHistPlot = myDataset.hist(column=myCol)
	plt.show()
'''
for feature in myDataframe.columns.values:
	viewDistribution(feature,myDataframe)
'''
# 6. When your data distributions are radically skewed, you can use a log scale to help reveal data that is otherwise too sparse to see. Write a new version of the viewDistribution function called viewLogDistribution to show the log distribution of each column. Add this function call after your viewDistribution call to view the regular and log distributions of each feature.

def viewLogDistribution(myCol,myDataset):
	myHistPlot = myDataset.hist(column=myCol, log=True)
	plt.show()
'''
for feature in myDataframe.columns.values:
	viewLogDistribution(feature,myDataframe)
'''
# 7. Use the two distributions to identify three bins per column that divide your data into roughly equal numbers. What are those bins? Note you do not need bins for "SeriousDlqin2yrs" as that is the feature you are modeling (it is your dependent variable).

def equalBins(myCol,myDataset):
	myBins = pd.qcut(myDataset[myCol], q=3, duplicates='drop', retbins=False).unique()
	return myBins
'''
for feature in myDataframe.columns.values:
	if feature != 'SeriousDlqin2yrs':
		equalBins(feature,myDataframe)
'''
# 8. Write a function called computeDefaultRisk that takes four arguments---a column name, a bin (as an array [start,end]), a target feature, and a dataframe---and returns the probability that someone will be at least 90 days delinquent on their account (in other words, "SeriousDlqin2yrs" = 1). Keep in mind that this probability is conditional, that means you'll want to use the equation for conditional probabilities to compute it. In plain English, you should compute the probability that a loan will become seriously delinquent given your target feature falls into the bin range. For example, if I'm looking at ages between 0 and 40, I want to compute the probability that a loan will go into serious delinquency given the applicant is between 0 and 40.

def bintoArray(myNum,myCol,myDataset):
	myBin = (equalBins(myCol,myDataset)[myNum].left,equalBins(myCol,myDataset)[myNum].right)
	return myBin

def computeDefaultRisk(myCol,binLoc,myFeature,myDataset):
	if binLoc == 'right':
		myNum = 0
	if binLoc == 'middle':
		myNum = 1
	if binLoc == 'left':
		myNum = 2
	count = 0
	count2 = 0
	try:
		myBin = bintoArray(myNum,myFeature,myDataset)	
	except:
		return 0.0
	for i, datapoint in myDataset.iterrows():
		if datapoint[myFeature] >= myBin[0] and datapoint[myFeature] < myBin[1]:
			count += 1
			if datapoint[myCol] == 1:
				count2 += 1
	totalSize = len(myDataset)
	prob = count / totalSize
	prob2 = count2 / totalSize
	finProb = prob2 / prob
	return finProb

# 9. Print out the risk of default for each of the feature bins in your dataset. Note it's helpful to label these with the feature and bins such that you can better understand your output.

myRisks = {}

for feature in myDataframe.columns.values:
	if feature != 'SeriousDlqin2yrs':
		featDict = {}
		featDict['left'] = (computeDefaultRisk('SeriousDlqin2yrs','left',feature,myDataframe)
		featDict['middle'] = computeDefaultRisk('SeriousDlqin2yrs','middle',feature,myDataframe)
		featDict['right'] = computeDefaultRisk('SeriousDlqin2yrs','right',feature,myDataframe)
		myRisks[feature] = featDict
'''
myRisks
'''
# 10. In your main file, use your loadAndCleanData function to load in newLoans.csv.

myLoanData = loadAndCleanData("newLoans.csv")

# 11. Use your conditional probabilities to predict the probability of default for each row in your CSV file. To do this, write a function called predictDefaultRisk that takes a row from your dataset as a parameter and returns the risk of default based on that data and the probabilities you computed from creditData.csv (hint: you might want to have predictDefaultRisk take a second parameter representing the risk of default for various data features computed from creditData.csv). You will want to compute the risk of default using a weighted sum with the following weights:

myWeights = {'age':0.025,'NumberOfDependents':0.025,'MonthlyIncome':0.1,'DebtRatio':0.1,'RevolvingUtilizationOfUnsecuredLines':0.1,'NumberOfOpenCreditLinesAndLoans':0.1,'NumberRealEstateLoansOrLines':0.1,'NumberOfTime30-59DaysPastDueNotWorse':0.15,'NumberOfTime60-89DaysPastDueNotWorse':0.15,'NumberOfTimes90DaysLate':0.15}

myBinDict = {}

for feature in myDataframe.columns:
	if feature != 'SeriousDlqin2yrs':
		mySideDict = {}
		try:
			mySideDict['right'] = equalBins(feature,myDataframe)[0]
		except:
			mySideDict['right'] = None
		try:
			mySideDict['middle'] = equalBins(feature,myDataframe)[1]
		except:
			mySideDict['middle'] = None
		try:
			mySideDict['left'] = equalBins(feature,myDataframe)[2]
		except:
			mySideDict['left'] = None
		myBinDict[feature] = mySideDict

myBinDict

def predictDefaultRisk(myRow,defaults,weights):
	probTable = []
	for feature in myRow.index:
		if feature != 'SeriousDlqin2yrs':
			# check which bin its in
			mybin = ''
			if myRow[feature] in myBinDict[feature]['right']:
				mybin = 'right'
			elif myRow[feature] in myBinDict[feature]['middle']:
				mybin = 'middle'
			elif myRow[feature] in myBinDict[feature]['left']:
				mybin = 'left'
			# get default prob and muliply it by weight
			myNum = defaults[feature][mybin] * weights[feature]
			# add to my probability sum table
			probTable.append(myNum)
	return sum(probTable)

# 12. Store the result of this function in the SeriousDlqin2yrs column. 

for row in range(len(myLoanData.index)):
	val = predictDefaultRisk(myLoanData.iloc[[row]],myRisks,myWeights)
	myLoanData['SeriousDlqin2yrs'][row] = val

# 13. Plot the distribution of risks using your computePDF function. What do you notice about this distribution?

computePDF('SeriousDlqin2yrs',testDF)

# It skews towards 0.