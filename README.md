# ECG-Signal-Segmentation-With-Hidden-Markov-Models
Using Hiddem Markov Model(HMM) to model the patterns in ECG signals in order to segment these signals into parts that are useful for further medical diagnosis. Figuring out these segments in each of the segments is often done by an human expert or automated systems that need to be hand-engineered and/or do not generalize well over variations in signal. We are exploring machine learning techniques - HMM in this one, to analyze how well they can perform and generalize on such task.

About Data:
Data being used is confidential so cannot be shared here.
ECG signal readings have been taken with 12 lead carodiogram method and with sampling rate for readings is 1000. Data has been truncated into equal sized ECG signals of length 800. 
As for in many applications, points of interest in ECG signal in our dataset are: 
'b'(i.e.baseline signal), 'p_onset', 'p_off', 'q_onset', 'r_peak', 's_end', 't_peak' and 't_end'.
Because of limitation of data being used, in this implementation, we are only ignoring detection of P segment from(p_onset to p_off). However, accomodating additional segment into the model later on is almost trivial.
Also, to model data in better way with HMM, we are cosidering signal to have states B1-QRS-B2-T-B3 rather than B-QRS-B-T-B, so as to simplify HMM architecture which can learn distribution in much better way.

Requirements: Python Libraries 
pomegranate 
Matplotlib
Numpy
pywt
sklearn

Results:
Simple Matching Coefficient percent accuracy on train set: 80.4071 %
Simple Matching Coefficient percent accuracy on test set: 79.2042 %
Dice index similarity measure for each of the segments:
B1: 0.969263
QRS: 0.742004
B3: 0.69598
T: 0.461033
B3: 0.695356

Future Improvements: Using Hidden Semi-Markov Models(HSMM) instead of standard HMM is a promising direction. Currently facing implementation issuees with the same. Repo will be updated once all issues in code with HSMM have been resolved.
