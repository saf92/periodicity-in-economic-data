#Import packages and functions
from packages import *
from functions import *


#Directory path which has the data
direc='/home/samuel/Documents/PhD/DSSG/data/'

#Directory to save plots in
direc_save='/home/samuel/Documents/PhD/DSSG/plots/'

#Add file names of nominal GDP (NGDP) and M4 money supply data
NGDP=direc+'UKNGDP.csv'
M4=direc+'UKM4.csv'

#M4 read data for 1963-2018
dataM4=pd.read_csv(M4)
dataM4=dataM4.values
xM4=dataM4[:,0]
yM4=dataM4[:,1]
yM4=yM4.astype(np.float)

dates=np.concatenate((xM4[0:24][::-1],xM4[24:len(xM4)][::-1]))

datesM4=np.arange(1963,2018.5,0.25)

M4v=np.concatenate((yM4[0:24][::-1],yM4[24:len(yM4)][::-1]))

#Plot the M4 quarterly data on a log scale
plt.figure(figsize=(15,10))
plt.plot(datesM4,M4v)
plt.title('M4 on a Log Scale',fontsize=30)
plt.xlabel('Date',fontsize=30)
plt.ylabel('£Million',fontsize=30)
plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.axvline(1974,color='black',linestyle='--')
plt.axvline(1990,color='black',linestyle='--')
plt.axvline(2010,color='black',linestyle='--')
plt.savefig(direc_save+'M4')
# plt.show()


#Make the rate of change of M4 time series data, note it's length is one less than the original data
datesM4roc=datesM4[1:len(datesM4)]
M4_rate_of_change=rate_of_change(M4v) #rate of change function in functions file


#Smooth the M4 rate of change with a four-fold running average, note this is 4 less than original data
M4roc_smooth=running_mean(M4_rate_of_change,4) #running mean function in function file
datesM4roc_smooth=datesM4roc[3:len(datesM4roc)]

#Plot of M4 rate of change and four-fold smoothed data
plt.figure(figsize=(15,10))
plt.plot(datesM4roc,M4_rate_of_change,label='Rate of change')
plt.plot(datesM4roc_smooth,M4roc_smooth,label='Smoothing')
plt.title(r'M4 Rate of Change: $\frac{M4_{t+1}}{M4_{t}}$',fontsize=30)
plt.xlabel('Date',fontsize=30)
plt.ylabel('Rate of Change',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.savefig(direc_save+'M4_roc')
# plt.show()

#Import adfuller function from package
# from statsmodels.tsa.stattools import adfuller

a1=adfuller(M4_rate_of_change)
a2=adfuller(M4roc_smooth)

# print(a1)
# print('')
# print(a2)

#Augmented Dickey Fuller test => both not stationary (second value is p-value)

#Define regression trees with max-depth 2 or 3
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2= DecisionTreeRegressor(max_depth=3)

#Reshape the rate of change smoothed data in form for regression tree function to work
xM4smooth=datesM4roc_smooth.reshape(-1,1)
yM4smooth=M4roc_smooth.reshape(-1,1)

#Fit on the smoothed data with both depths
regr_1.fit(xM4smooth,yM4smooth)
regr_2.fit(xM4smooth,yM4smooth)

#Predict on the same data to get values for regression tree
y_1M4smooth = regr_1.predict(xM4smooth)
y_2M4smooth = regr_2.predict(xM4smooth)

#Plot the smoothed data and the regression trees with max depth 2 and 3
plt.figure(figsize=(15,10))
plt.plot(xM4smooth,yM4smooth,'o')
plt.plot(xM4smooth,y_1M4smooth,linewidth=3,label='Max depth=2')
plt.plot(xM4smooth,y_2M4smooth,linewidth=3,label='Max depth=3')
plt.title('Regression Tree M4 Smoothing',fontsize=30)
plt.xlabel('Date',fontsize=30)
plt.ylabel('Rate of Change',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.axvline(1974,color='black',linestyle='--')
plt.axvline(1990,color='black',linestyle='--')
plt.axvline(2008,color='black',linestyle='--')
plt.legend(fontsize=20)
plt.savefig(direc_save+'M4_roc_trees')


#Test for overfitting
X=xM4smooth
y=yM4smooth

#Split data randomly into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=51)

#Predictions on training set
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2= DecisionTreeRegressor(max_depth=3)

regr_1.fit(X_train,y_train)
regr_2.fit(X_train,y_train)


predictions1tr=regr_1.predict(X_train)
predictions2tr=regr_2.predict(X_train)

print(mean_squared_error(predictions1tr,y_train)) #Note very small mean squared error due
#to small differences in data values
print(mean_squared_error(predictions2tr,y_train))

#Mean squared error for prediction on training and test sets after fitting on training data
#for different depths of tree
MSE_train=[]
MSE_test=[]
for k in range(1,10):
    regr = DecisionTreeRegressor(max_depth=k)
    regr.fit(X_train,y_train)
    predictions_train=regr.predict(X_train)
    predictions_test=regr.predict(X_test)
    mse_train=mean_squared_error(predictions_train,y_train)
    mse_test=mean_squared_error(predictions_test,y_test)
    MSE_train.append(mse_train)
    MSE_test.append(mse_test)

#Plot of MSE of predictions on training and testing set
plt.figure(figsize=(15,10))
r=range(1,10)
plt.plot(r,MSE_train,'o',label='train')
plt.plot(r,MSE_test,'o',label='test')

plt.title(r"MSE of predictions of training and testing M4" "\n" r"with increasing tree depth",fontsize=30)
plt.xlabel('Depth',fontsize=30)
plt.ylabel('MSE',fontsize=30)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)

plt.savefig(direc_save+'MSE_M4')

from sklearn.model_selection import KFold
#K-fold cross validation

#10 folds
kf = KFold(n_splits=10)

s=kf.get_n_splits(X)

t=10

#Define matrix to hold MSE: rows are for the depth, columns for the split
MSE=np.zeros((t,s))

for k in range(1,t+1):
    count=0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regr = DecisionTreeRegressor(max_depth=k)
        regr.fit(X_train,y_train)
        prediction = regr.predict(X_test)
        mse=mean_squared_error(prediction,y_test)
        MSE[k-1,count]=mse
        count+=1

#Find the mean of the MSEs of the depths of each of the rows
MSE_mean=[]

for i in range(t):
    MSE_mean.append(np.mean(MSE[i,:]))

#Find the depth with minimum mean squared error, note dependant on number of folds
a=np.asarray(MSE_mean,float)
np.where(a == a.min())

mymin = np.min(a)
min_positions = [i for i, x in enumerate(a) if x == mymin]

m=min_positions[0]+1

print(m)


#Plot of 10-fold MSE on testing set
plt.figure(figsize=(15,10))
r=range(1,t+1)
plt.plot(r,MSE_mean)

plt.title(r"Average of MSE of predictions of training on 10-fold testing M4" "\n" r"with increasing tree depth",fontsize=30)
plt.xlabel('Depth',fontsize=30)
plt.ylabel('MSE',fontsize=30)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.savefig(direc_save+'MSE_k_fold_M4')

#Hodrick-Prescott filter M4 ####################################################################

cycleM4roc, trendM4roc = sm.tsa.filters.hpfilter(M4roc_smooth, 1600) #finds cyclical and trend component
#using HP filter to get cycle and trend component

#Augmented Dickey Fuller test, p-value <0.01 => accept null that data is stationary
print(adfuller(cycleM4roc))

#Periodogram of rate of change of M4 of cyclic component
plt.figure(figsize=(15,10))
p=periodogram(cycleM4roc)

q=int((len(p)-1)/2)
plt.plot(range(1,q),p[1:q])
plt.title('Periodogram for Hodrick-Prescott Transformed M4 ROC',fontsize=30)
plt.xlabel('Frequency',fontsize=30)

plt.axvline(6,color='black',linestyle='--')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.text(5,-0.00008,'6',fontsize=30)

plt.savefig(direc_save+'M4_HP_periodogram')

#Finish HP-Filter ########################################################################

#Transform M4 roc data by dividing by decision tree with max-depth 2
M4roc_transform1=M4roc_smooth/y_1M4smooth

#Second transform M4 roc data by taking logs of previous transform
M4roc_transform2=np.log(M4roc_smooth/y_1M4smooth)

#Augmented Dickey-Fuller test on first transform
print(adfuller(M4roc_transform1))

#Periodogram of transformed M4 data
plt.figure(figsize=(15,10))
p=periodogram(M4roc_transform1)

q=int((len(p)-1)/2)# Periodogram symmetric, plot only first half
plt.plot(range(1,q),p[1:q]) #Don't include 0th frequency as it is so comparably large and does not give
#relevant info
plt.title('Periodogram for Transformed M4',fontsize=30)
plt.xlabel('Frequency',fontsize=30)

plt.axvline(6,color='black',linestyle='--')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.text(5,-0.00014,'6',fontsize=30)

plt.savefig(direc_save+'M4_periodogram_frequency')

#Fit periods based on the frequencies (divide total length of years by frequency)

r=xM4smooth[len(xM4smooth)-1]-xM4smooth[0]
r=r[0]
periodM4=[]

for i in range(1,q):
    periodM4.append(r/i)

periodM4h=round(periodM4[5],2)

print(periodM4h) #Period with frequency 6

#Plot of periodogram with periods
plt.figure(figsize=(15,10))
p=periodogram(M4roc_transform1)

q=int((len(p)-1)/2)
plt.plot(periodM4,p[1:q])
plt.title('Periodogram for Transformed M4',fontsize=30)
plt.xlabel('Period (Years)',fontsize=30)


plt.axvline(periodM4[5],color='black',linestyle='--')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.text(9,0,'{c}'.format(c=periodM4h),fontsize=30)

plt.savefig(direc_save+'M4_periodogram_period')

#Find the beginning of each new cycle with period 6
periods=[]
years=[]
for i in range(1,6):
    per=xM4smooth[0][0]+i*periodM4[5]
    y=i*periodM4[5]
    periods.append(per)
    years.append(y)

plt.figure(figsize=(15,10))
x=xM4smooth
y=M4roc_transform1

n = len(y)
Y=np.fft.fft(y)


for i in range(len(periods)):
    plt.axvline(periods[i],color='black',linestyle='--')

c=6

#Inverse Fourier transform of first 6 componenets
np.put(Y, range(c+1, n), 0.0)

ifft=np.fft.ifft(Y)

plt.plot(x,y)
plt.plot(x,ifft)

plt.title('{c} Fourier Components of Transformed M4 Rate of Change'.format(c=c),fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.axvline(per)
plt.savefig(direc_save+'M4_inv_period')

#Find the time in years starting with first data value at 0
xs=np.arange(0,(len(xM4smooth)-1)/4,0.25)

#Plot of correlogram for M4 transformed rate of change
plt.figure(figsize=(15,10))
plt.plot(xs,AUT1(M4roc_transform1))
plt.title('Correlogram for M4 Transformed Rate of Change',fontsize=30)
plt.ylabel(r'$r_k$',fontsize=30)
plt.xlabel('Years',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

for i in range(len(years)):
    plt.axvline(years[i],color='black',linestyle='--')
plt.savefig(direc_save+'M4_cor')


###########################################################################################################
#Do same for NGDP
###########################################################################################################
dataNGDP=pd.read_csv(NGDP)
dataNGDP=dataNGDP.values
NGDPx=dataNGDP[:,0] #dates 1955-2018, quarterly, strings
NGDPy=dataNGDP[:,1]
NGDP=NGDPy.astype(np.float) #NGDP as float

datesNGDP=np.arange(1955,2018.25,0.25)

plt.figure(figsize=(15,10))
plt.plot(datesNGDP,NGDP)
plt.title('Nominal Quarterly GDP',fontsize=30)
plt.xlabel('Date',fontsize=30)
plt.ylabel('£Million',fontsize=30)
plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.axvline(1974,color='black',linestyle='--')
plt.axvline(1990,color='black',linestyle='--')
plt.axvline(2008,color='black',linestyle='--')
plt.savefig(direc_save+'NGDP')

NGDP_rate_of_change=rate_of_change(NGDP)
datesNGDProc=datesNGDP[1:len(datesNGDP)]

NGDProc_smooth=running_mean(NGDP_rate_of_change,4)
datesNGDProc_smooth=datesNGDProc[3:len(datesNGDProc)]

plt.figure(figsize=(15,10))
plt.plot(datesNGDProc,NGDP_rate_of_change,label='Rate of change')
plt.plot(datesNGDProc_smooth,NGDProc_smooth,label='Smoothing')
plt.title(r'NGDP Rate of Change: $\frac{NGDP_{t+1}}{NGDP_{t}}$',fontsize=30)
plt.xlabel('Date',fontsize=30)
plt.ylabel('Rate of Change',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.savefig(direc_save+'NGDP_roc')

n1=adfuller(NGDP_rate_of_change)
n2=adfuller(NGDProc_smooth)

print(n1)
print('')
print(n2)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2= DecisionTreeRegressor(max_depth=3)

xNGDPsmooth=datesNGDProc_smooth.reshape(-1,1)
yNGDPsmooth=NGDProc_smooth.reshape(-1,1)

regr_1.fit(xNGDPsmooth,yNGDPsmooth)
regr_2.fit(xNGDPsmooth,yNGDPsmooth)

y_1NGDPsmooth = regr_1.predict(xNGDPsmooth)
y_2NGDPsmooth = regr_2.predict(xNGDPsmooth)

plt.figure(figsize=(15,10))
plt.plot(xNGDPsmooth,yNGDPsmooth,'o')
plt.plot(xNGDPsmooth,y_1NGDPsmooth,linewidth=3,label='Max depth=2')
plt.plot(xNGDPsmooth,y_2NGDPsmooth,linewidth=3,label='Max depth=3')
plt.title('Regression Tree Smooth NGDP',fontsize=30)
plt.xlabel('Date',fontsize=30)
plt.ylabel('Rate of Change',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.axvline(1974,color='black',linestyle='--')
plt.axvline(1990,color='black',linestyle='--')
plt.axvline(2008,color='black',linestyle='--')
plt.legend(fontsize=20)
plt.savefig(direc_save+'NGDP_roc_trees')

X=xNGDPsmooth
y=yNGDPsmooth

#Split data randomly into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=51)

#Predictions on training set
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2= DecisionTreeRegressor(max_depth=3)

regr_1.fit(X_train,y_train)
regr_2.fit(X_train,y_train)


predictions1tr=regr_1.predict(X_train)
predictions2tr=regr_2.predict(X_train)

print(mean_squared_error(predictions1tr,y_train)) #Note very small mean squared error due
#to small differences in data values
print(mean_squared_error(predictions2tr,y_train))

#Mean squared error for prediction on training and test sets after fitting on training data
#for different depths of tree
MSE_train=[]
MSE_test=[]
for k in range(1,10):
    regr = DecisionTreeRegressor(max_depth=k)
    regr.fit(X_train,y_train)
    predictions_train=regr.predict(X_train)
    predictions_test=regr.predict(X_test)
    mse_train=mean_squared_error(predictions_train,y_train)
    mse_test=mean_squared_error(predictions_test,y_test)
    MSE_train.append(mse_train)
    MSE_test.append(mse_test)

plt.figure(figsize=(15,10))
r=range(1,10)
plt.plot(r,MSE_train,'o',label='train')
plt.plot(r,MSE_test,'o',label='test')

plt.title(r"MSE of predictions of training and testing NGDP" "\n" r"with increasing tree depth",fontsize=30)
plt.xlabel('Depth',fontsize=30)
plt.ylabel('MSE',fontsize=30)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)

plt.savefig(direc_save+'MSE_NGDP')

kf = KFold(n_splits=10)

s=kf.get_n_splits(X)

t=10

#Define matrix to hold MSE: rows are for the depth, columns for the split
MSE=np.zeros((t,s))

for k in range(1,t+1):
    count=0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regr = DecisionTreeRegressor(max_depth=k)
        regr.fit(X_train,y_train)
        prediction = regr.predict(X_test)
        mse=mean_squared_error(prediction,y_test)
        MSE[k-1,count]=mse
        count+=1

#Find the mean of the MSEs of the depths of each of the rows
MSE_mean=[]

for i in range(t):
    MSE_mean.append(np.mean(MSE[i,:]))

#Find the depth with minimum mean squared error, note dependant on number of folds
a=np.asarray(MSE_mean,float)
np.where(a == a.min())

mymin = np.min(a)
min_positions = [i for i, x in enumerate(a) if x == mymin]

m=min_positions[0]+1

print(m)

plt.figure(figsize=(15,10))
r=range(1,t+1)
plt.plot(r,MSE_mean)

plt.title(r"Average of MSE of predictions of training on 10-fold testing NGDP" "\n" r"with increasing tree depth",fontsize=30)
plt.xlabel('Depth',fontsize=30)
plt.ylabel('MSE',fontsize=30)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.savefig(direc_save+'MSE_k_fold_NGDP')

NGDProc_transform=NGDProc_smooth/y_2NGDPsmooth
NGDProc_transform1=NGDProc_smooth-y_2NGDPsmooth
NGDProc_transform2=NGDProc_smooth-np.mean(NGDProc_smooth)

cycleNGDProc, trendNGDProc = sm.tsa.filters.hpfilter(NGDProc_smooth, 1600)

adfuller(cycleNGDProc)

plt.figure(figsize=(15,10))
p1=periodogram(cycleNGDProc)

q1=int((len(p1)-1)/2)
plt.plot(range(1,q1),p1[1:q1])
plt.title('Periodogram for Hodrick Prescott NGDP ROC',fontsize=30)
plt.xlabel('Frequency',fontsize=30)

plt.axvline(12,color='black',linestyle='--')
plt.axvline(16,color='black',linestyle='--')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.savefig(direc_save+'NGDP_HP_periodogram')

plt.figure(figsize=(15,10))
p1=periodogram(NGDProc_transform)

q1=int((len(p1)-1)/2)
plt.plot(range(1,q1),p1[1:q1])
plt.title('Periodogram for Transformed NGDP',fontsize=30)
plt.xlabel('Frequency',fontsize=30)

plt.axvline(10,color='black',linestyle='--')
plt.axvline(16,color='black',linestyle='--')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.text(6,-0.00002,'10',fontsize=30)
plt.savefig(direc_save+'NGDP_periodogram_frequency',bbox_inches='tight')

n2=len(NGDProc_transform)
r1=xNGDPsmooth[len(xNGDPsmooth)-1]-xNGDPsmooth[0]
r1=r1[0]
periodNGDP=[]

for i in range(1,q1):
    periodNGDP.append(r1/i)

periodNGDPh=round(periodNGDP[9],2)

plt.figure(figsize=(15,10))
p1=periodogram(NGDProc_transform)

q1=int((len(p1)-1)/2)
plt.plot(periodNGDP,p1[1:q1])
plt.title('Periodogram for Transformed NGDP',fontsize=30)
plt.xlabel('Period (Years)',fontsize=30)

plt.axvline(periodNGDP[9],color='black',linestyle='--')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.text(6,0,'{c}'.format(c=periodNGDPh),fontsize=30)

plt.savefig(direc_save+'NGDP_periodogram_period',bbox_inches='tight')

periodNGDPh=round(periodNGDP[9],2)

periods1=[]

years1=[]

for i in range(1,10):
    per1=xNGDPsmooth[0][0]+periodNGDP[9]*i
    y1=periodNGDP[9]*i
    periods1.append(per1)
    years1.append(y1)

plt.figure(figsize=(15,10))
x1=xNGDPsmooth
y1=NGDProc_transform

n = len(y1)
Y1=np.fft.fft(y1)

for i in range(len(periods1)):
    plt.axvline(periods1[i],color='black',linestyle='--')



c=10
#np.put(Y1, range(0, c), 0.0)
np.put(Y1, range(c+1, n), 0.0)

ifft1=np.fft.ifft(Y1)

plt.plot(x1,y1)
plt.plot(x1,ifft1)

plt.title('{c} Fourier Components of Transformed NGDP Rate of Change'.format(c=c),fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.savefig(direc_save+'NGDP_inv_period')

xl=np.arange(0,(len(xNGDPsmooth)-1)/4,0.25)

plt.figure(figsize=(15,10))
plt.plot(xl,AUT1(NGDProc_transform))

for i in range(len(years1)):
    plt.axvline(years1[i],color='black',linestyle='--')
plt.title('Correlogram for NGDP Transformed Rate of Change',fontsize=30)
plt.ylabel(r'$r_k$',fontsize=30)
plt.xlabel('Years',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig(direc_save+'NGDP_cor')

plt.figure(figsize=(15,10))
plt.plot(x,ifft,label='M4 6 Components')
plt.plot(x1,ifft1,label='NGDP 10 Components')

plt.title('Inverse Fourier Transform of M4 and NGDP',fontsize=30)
plt.ylabel('',fontsize=30)
plt.xlabel('Years',fontsize=30)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.savefig(direc_save+'M4_NGDP_Inv_period')
