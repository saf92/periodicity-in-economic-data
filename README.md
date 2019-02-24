# periodicity-in-economic-data

Python3 code that 

1) makes UK money supply and GDP growth data stationary by detrending the data with decision tree regression. 

2) finds the periodogram of the detrended data, indicating a significant frequency of 6 corresponding to a period of roughly 9 years for money supply (M4) data. The frequency results were less obvious for GDP growth but indication of higher frequencies and smaller periods.

3) gives a comparison of this method of detrending to the more standard Hodrick-Prescott filter. Both methods give the same result for money supply data and indication of smaller less pronounced frequencies for GDP data as before.

4) finds the inverse Fourier transform on the number of Fourier components equal to the highest frequency in periodogram and overlays this on the growth data.

5) finds the correlogram of the transformed data indicating for money supply data the same frequency and the smaller frequencies for GDP as before.
