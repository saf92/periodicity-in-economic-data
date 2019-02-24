
import numpy as np

#function to find first differences in an vector

def time_dif(x):
    t=[]
    for i in range(len(x)-1):
        d=x[i+1]-x[i]
        t.append(d)
    return t

#finds rate of change of time series
def rate_of_change(x):
    rates_vector=[]
    for i in range(len(x)-1):
        r=x[i+1]/x[i]
        rates_vector.append(r)
    rates_vector=np.asarray(rates_vector,float)
    return rates_vector

#two functions that finds moving average of length N
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def running_mean1(x,N):
    r=np.convolve(x, np.ones((N,))/N, mode='valid')
    return r

#Autocorrelation coefficient with lag time k

def auto_cor(x,k):
    x_bar=np.mean(x)
    N=len(x)
    z1=0
    z2=0
    for i in range(N):
        z1=z1+(x[i]-x_bar)**2
        
    for i in range(N-k):
        z2=z2+(x[i]-x_bar)*(x[i+k]-x_bar)
        
    r_k=z2/z1
    
    return r_k

#Autocorrelations for lags k=0,..,N/4

def AUT(x):
    N=len(x)
    AUT=[]
    for k in range(int(N/4)+1):
        AUT.append(auto_cor(x,k))
        
    return AUT

#Autocorrelations for all lags k=0,...,N-1

def AUT1(x):
    N=len(x)
    AUT=[]
    for k in range(N-1):
        AUT.append(auto_cor(x,k))
        
    return AUT

#Defining the periodogram with a function
def periodogram(x):
    N=len(x)
    Y=np.fft.fft(x)
    p=1/N*np.absolute(Y)**2
    return p


