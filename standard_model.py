'''
Code uses standard trained model for prediction on transformed wavelet.
Uses MAE to calculate accuracy. 
'''

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import seaborn as sns
from pomegranate import *
import pywt
from matplotlib.pyplot import xlabel
from sklearn import model_selection
from sklearn.preprocessing.tests import test_label
import random

def transform(data):
    temp = data[0][0:800]
    for i,item in enumerate(data):
        #ret = pywt.swt(data=item[0:800],wavelet="db1",level=2)[0][0]
        data[i] = pywt.swt(data=item[0:800],wavelet="db1",level=4)[0][0]
    return data

def trainModel(seq,label,model=None):
    print("Starting with building the model")
    if model is None:
        model = HiddenMarkovModel()
        ret_model = model.from_samples(NormalDistribution  , \
        n_components=5, X=seq, labels=label, algorithm='labeled',\
        verbose=True,min_iterations=10)
        
        ret_model.freeze()
        ret_model.bake()
        
        return ret_model
    else:
        model = model.fit(seq,label,len(seq))
    return model


def extract_points(master_record, master_label, master_file_list, log):
    p_onset, q_onset, r_peak, s_end, t_peak, t_end = \
        ([] for i in range(6))
    p_onset_loc, q_onset_loc, r_peak_loc, s_end_loc, t_peak_loc, \
        t_end_loc = ([] for i in range(6))
    record_missing_labels = []
    
    for labelset, recordset, file in zip(master_label, master_record,\
                                    master_file_list): 
        if 'p_onset' not in labelset:
            record_missing_labels.append([file, 'p_onset'])
            log.write("'p_onset' is not in file: " + file  + "\n")
        else:
            p_onset.append(recordset[labelset.index("p_onset")])
            p_onset_loc.append(labelset.index("p_onset"))
        
        if 'q_onset' not in labelset:
            record_missing_labels.append([file, 'q_onset'])
            log.write("'q_onset' is not in file: " + file  + "\n")
        else:
            q_onset.append(recordset[labelset.index("q_onset")])
            q_onset_loc.append(labelset.index("q_onset"))
        
        if 'r_peak' not in labelset:
            record_missing_labels.append([file, 'r_peak'])
            log.write("'r_peak' is not in file: " + file  + "\n")
        else:
            r_peak.append(recordset[labelset.index("r_peak")])
            r_peak_loc.append(labelset.index("r_peak"))
        
        if 's_end' not in labelset:
            record_missing_labels.append([file, 's_end'])
            log.write("'s_end' is not in file: " + file + "\n")
        else:
            s_end.append(recordset[labelset.index("s_end")])
            s_end_loc.append(labelset.index("s_end"))
        
        if 't_peak' not in labelset:
            record_missing_labels.append([file, 't_peak'])
            log.write("'t_peak' is not in file: " + file  + "\n")
        else:
            t_peak.append(recordset[labelset.index("t_peak")])
            t_peak_loc.append(labelset.index("t_peak"))
        
        if 't_end' not in labelset:
            record_missing_labels.append([file, 't_end'])
            log.write("'t_end' is not in file: " + file  + "\n")
        else:
            t_end.append(recordset[labelset.index("t_end")])
            t_end_loc.append(labelset.index("t_end"))
    all_points = [p_onset, p_onset_loc, q_onset, q_onset_loc, r_peak, \
                  r_peak_loc, s_end, s_end_loc, t_peak, t_peak_loc, \
                  t_end, t_end_loc]
    
    return record_missing_labels, all_points

def get_mean_sd(curr_list):
    mean = np.mean(curr_list)    
    sd = np.std(curr_list)
    return mean, sd

def handle_missing_labels(record_missing_labels, master_record,\
    master_label, all_points, master_file_index, master_file_list, log):
    p_onset_loc = all_points[1] 
    q_onset_loc = all_points[3] 
    r_peak_loc = all_points[5] 
    s_end_loc = all_points[7]
    t_peak_loc = all_points[9] 
    t_end_loc = all_points[11]
    p_mean, p_sd = get_mean_sd(p_onset_loc)
    q_mean, q_sd = get_mean_sd(q_onset_loc)
    r_mean, r_sd = get_mean_sd(r_peak_loc)
    s_mean, s_sd = get_mean_sd(s_end_loc)
    t_mean, t_sd = get_mean_sd(t_peak_loc)
    tend_mean, tend_sd = get_mean_sd(t_end_loc)
    
    for item in record_missing_labels:
        if item[1] == 'p_onset':
            ind = master_file_index[item[0]]
            master_label[ind][int(p_mean)] = 'p_onset'
        elif item[1] == 'q_onset':
            ind = master_file_index[item[0]]
            master_label[ind][int(q_mean)] = 'q_onset'
        elif item[1] == 'r_peak':
            ind = master_file_index[item[0]]
            master_label[ind][int(r_mean)] = 'r_peak'
        elif item[1] == 's_end':
            ind = master_file_index[item[0]]
            master_label[ind][int(s_mean)] = 's_end'
        elif item[1] == 't_peak':
            ind = master_file_index[item[0]]
            master_label[ind][int(t_mean)] = 't_peak'
        elif item[1] == 't_end':
            ind = master_file_index[item[0]]
            master_label[ind][int(tend_mean)] = 't_end'
    
    return copy(master_label)
    
def get_outliers_loc(curr_list):
    outliers = []
    curr_mean , curr_sd = get_mean_sd(curr_list) 
    for item in curr_list:
        if not (item >= (curr_mean - 3*curr_sd) and 
                item <= (curr_mean + 3*curr_sd)):
            outliers.append(item)
    return outliers

def get_percentile(curr_list, number):
    return np.percentile(curr_list, number)

def correct_outliers_loc(master_record, master_label, master_file_list,\
                     master_file_index, log):
    record_missing_labels, all_points = extract_points(master_record, \
                master_label, master_file_list, log)
    locs = [1, 3, 5, 7, 9, 11]
    labels = ['p_onset','q_onset','r_peak','s_end','t_peak','t_end']
    means = {}
    p5 = {}
    p95 = {}
    #Understand data parameters of each label's distribution
    for label, loc in zip(labels, locs): 
        mean, sd = get_mean_sd(all_points[loc])
        means[loc] = mean
        p5[label] = int(get_percentile(all_points[loc], 5))
        p95[label] = int(get_percentile(all_points[loc], 95))
    #Correct outliers to set mean location as new label location.\
    #And set old location's label to 'B'
    count = 0
    for list in master_label:
        for label in labels:
            curr_loc = list.index(label)
            if curr_loc < p5[label]:
                list[p5[label]] = label
                list[curr_loc] = 'B'
                count = count + 1
            elif curr_loc > p95[label]:
                list[p95[label]] = label
                list[curr_loc] = 'B'
                count = count + 1
    print("Number of outliers corrected: ", count)        
    #sanity check: make sure this label replacement did not cause any\
    #loss of labels 
    record_missing_labels, all_points = extract_points(master_record, \
                master_label, master_file_list, log)
    print("Sanity check: Number of missing labels after correcting outliers: ", len(record_missing_labels))
    
    return master_label

def plot_data(master_record, master_label, master_file_list, log):
    record_missing_labels, all_points = extract_points(master_record, \
                master_label, master_file_list, log)
    plt.xlabel("Location in truncated reading data")
    plt.ylabel("Reading amplitude AND Frequency for histogram")
    plt.scatter(all_points[1],np.array(all_points[0])*100, color='red', label='p_onset')
    plt.scatter(all_points[3],np.array(all_points[2])*100, color='green', label='q_onset')
    plt.scatter(all_points[5],np.array(all_points[4])*100, color='blue', label='r_peak')
    plt.scatter(all_points[7],np.array(all_points[6])*100, color='yellow', label='s_end')
    plt.scatter(all_points[9],np.array(all_points[8])*100, color='pink', label='t_peak')
    plt.scatter(all_points[11],np.array(all_points[10])*100, color='black', label='t_end')
    plt.legend()
    plt.show()

def plot_dist(master_record, master_label, master_file_list, log):
    record_missing_labels, all_points = extract_points(master_record, \
                master_label, master_file_list, log)
    #sns.distplot(all_points[1], 20, hist=False,rug=True,label='p_onset_loc')
    #sns.distplot(all_points[3], 20, hist=False,rug=True,label='q_onset_loc')
    
    #does not get plot since 0 variance.
    #sns.distplot(r_peak_loc, 20, hist=False,rug=True)
    
    #sns.distplot(all_points[7], 20, hist=False,rug=True,label='s_end_loc')
    #sns.distplot(all_points[9], 20, hist=False,rug=True,label='t_peak_loc')
    #sns.distplot(all_points[11], 20, hist=False,rug=True,label='t_end_loc')
    #plt.xlabel("Distribution of points along the ECG location")
    #plt.legend()
    #plt.show()
    B1_loc, B2_loc, B3_loc, B4_loc, B5_loc, B6_loc, B7_loc, p_onset_loc,\
    q_onset_loc, r_peak_loc, s_end_loc, t_peak_loc, t_end_loc = \
        ([] for i in range(13))
    print("I am here")
    count = 1
    for curr_list in master_label:
        print("Processing list number: ", count)
        for index, label in enumerate(curr_list):
            if label == "B1":
                B1_loc.append(index) 
            elif label == "B2":
                B2_loc.append(index) 
            elif label == "B3":
                B3_loc.append(index)
            elif label == "B4":
                B4_loc.append(index)
            elif label == "B5":
                B5_loc.append(index)
            elif label == "B6":
                B6_loc.append(index)
            elif label == "B7":
                B7_loc.append(index) 
            elif label == "p_onset":
                p_onset_loc.append(index)
            elif label == "q_onset":
                q_onset_loc.append(index)
            elif label == "r_peak":
                r_peak_loc.append(index)
            elif label == "s_end":
                s_end_loc.append(index)
            elif label == "t_peak":
                t_peak_loc.append(index)
            elif label == "t_end":
                t_end_loc.append(index)
        count = count + 1
    distributions = [B1_loc, B2_loc, B3_loc, B4_loc, B5_loc,\
        B6_loc, B7_loc,p_onset_loc, q_onset_loc, r_peak_loc,\
        s_end_loc,t_peak_loc, t_end_loc]
    for dist in distributions:
        sns.distplot(dist, 20, hist=False,rug=True,label='dist')
    plt.legend()
    plt.show()

def update_labels_to_digits(master_label):
    mapping = {"B1":"s0","QRS":"s1","B2":"s2","T":"s3","B3":"s4"}
    for curr_list  in master_label:
        ind = 0
        for ind in range(0, len(curr_list)):
            curr_list[ind] = mapping[curr_list[ind]]
    return master_label


def create_label_blocks(master_label):
    diff = []
    for list1 in master_label:
        start = list1.index("q_onset")
        end = list1.index("s_end")
        for i in range(0, start):
            list1[i] = "B1"
        for i in range(start,end+1):
            list1[i] = "QRS"
        start = list1.index("t_peak")
        endnew = list1.index("t_end")
        diff.append(endnew-end)
        for i in range(end+1, start):
            list1[i] = "B2"
        for i in range(start, endnew+1):
            list1[i] = "T"
        for i in range(endnew+1,len(list1)):
            list1[i] = "B3"
    return master_label, diff

def generate_loc(start, end):
    loc = []
    for i in range(start, end):
        loc.append(i)
    return loc

def new_plot1(all_test_predictions, test_record, orig_test_record,\
              test_label, test_files, num_iterations):
    for i in range(0, num_iterations):
        curr_pred_list = all_test_predictions[i]
        for prediction, record, orig_record, curr_test_label, filename in \
        zip(curr_pred_list, test_record, orig_test_record, test_label,\
        test_files):
            if((0 not in prediction) or (1 not in prediction) or \
            (2 not in prediction) or (3 not in prediction) or \
            (4 not in prediction)):
                print("Skipped while plotting???")
                #return 0
                continue
            else:
                B1 = record[0 : prediction.index(1)]
                B1_loc = generate_loc(0, prediction.index(1))
    
                QRS = record[prediction.index(1) : prediction.index(2)]
                QRS_loc = generate_loc(prediction.index(1), prediction.index(2))
    
                B2 = record[prediction.index(2) : prediction.index(3)]
                B2_loc = generate_loc(prediction.index(2), prediction.index(3))
    
                T = record[prediction.index(3) : prediction.index(4)]
                T_loc = generate_loc(prediction.index(3), prediction.index(4))
    
                B3 = record[prediction.index(4) : len(prediction)]
                B3_loc = generate_loc(prediction.index(4), len(prediction))
    
                plt.plot(B1_loc,B1,color="red",label="B1")
                plt.plot(QRS_loc, QRS, color="green",label="QRS")
                plt.plot(B2_loc,B2,color="blue",label="B2")
                plt.plot(T_loc,T,color="orange",label="T")
                plt.plot(B3_loc,B3,color="black",label="B3")
                
                #-------------------------------------------------------
                B1t = orig_record[0 : curr_test_label.index(1)]
                B1t_loc = generate_loc(0, curr_test_label.index(1))
    
                QRSt = orig_record[curr_test_label.index(1) : curr_test_label.index(2)]
                QRSt_loc = generate_loc(curr_test_label.index(1), curr_test_label.index(2))
    
                B2t = orig_record[curr_test_label.index(2) : curr_test_label.index(3)]
                B2t_loc = generate_loc(curr_test_label.index(2), curr_test_label.index(3))
    
                Tt = orig_record[curr_test_label.index(3) : curr_test_label.index(4)]
                Tt_loc = generate_loc(curr_test_label.index(3), curr_test_label.index(4))
    
                B3t = orig_record[curr_test_label.index(4) : len(curr_test_label)]
                B3t_loc = generate_loc(curr_test_label.index(4), len(curr_test_label))
    
                plt.plot(B1t_loc,B1t,color="red",label="B1t")
                plt.plot(QRSt_loc, QRSt, color="green",label="QRSt")
                plt.plot(B2t_loc,B2t,color="blue",label="B2t")
                plt.plot(Tt_loc,Tt,color="orange",label="Tt")
                plt.plot(B3t_loc,B3t,color="black",label="B3t")
    
                plt.title("5 states")
                plt.savefig("F:\\Courses\\IndStudy\\My_Work\\STANDARD_HMM_5_STATES\\"\
                            +filename+"_iteration"+str(i)+".png",bbox_inches="tight")
                plt.close()
                    
    
def new_plot(prediction, record, orig_record, temp_label, filename):
    plt.figure()
    if((0 not in prediction) or (1 not in prediction) or (2 not in prediction) or (3 not in prediction) or (4 not in prediction)):
        print("Skipped while plotting???")
        return 0
    else:
        B1 = record[0 : prediction.index(1)]
        B1_loc = generate_loc(0, prediction.index(1))
    
        QRS = record[prediction.index(1) : prediction.index(2)]
        QRS_loc = generate_loc(prediction.index(1), prediction.index(2))
    
        B2 = record[prediction.index(2) : prediction.index(3)]
        B2_loc = generate_loc(prediction.index(2), prediction.index(3))
    
        T = record[prediction.index(3) : prediction.index(4)]
        T_loc = generate_loc(prediction.index(3), prediction.index(4))
    
        B3 = record[prediction.index(4) : len(prediction)]
        B3_loc = generate_loc(prediction.index(4), len(prediction))
    
        plt.plot(B1_loc,B1,color="red",label="B1")
        plt.plot(QRS_loc, QRS, color="green",label="QRS")
        plt.plot(B2_loc,B2,color="blue",label="B2")
        plt.plot(T_loc,T,color="orange",label="T")
        plt.plot(B3_loc,B3,color="black",label="B3")
        #-----------------------------------------------------------
        B1t = orig_record[0 : temp_label.index(1)]
        B1t_loc = generate_loc(0, temp_label.index(1))
    
        QRSt = orig_record[temp_label.index(1) : temp_label.index(2)]
        QRSt_loc = generate_loc(temp_label.index(1), temp_label.index(2))
    
        B2t = orig_record[temp_label.index(2) : temp_label.index(3)]
        B2t_loc = generate_loc(temp_label.index(2), temp_label.index(3))
    
        Tt = orig_record[temp_label.index(3) : temp_label.index(4)]
        Tt_loc = generate_loc(temp_label.index(3), temp_label.index(4))
    
        B3t = orig_record[temp_label.index(4) : len(temp_label)]
        B3t_loc = generate_loc(temp_label.index(4), len(temp_label))
    
        plt.plot(B1t_loc,B1t,color="red",label="B1t")
        plt.plot(QRSt_loc, QRSt, color="green",label="QRSt")
        plt.plot(B2t_loc,B2t,color="blue",label="B2t")
        plt.plot(Tt_loc,Tt,color="orange",label="Tt")
        plt.plot(B3t_loc,B3t,color="black",label="B3t")
    
        plt.title("5 states")
        plt.savefig("F:\\Courses\\IndStudy\\My_Work\\TEST5\\"+filename+".png",bbox_inches="tight")
        plt.close()
        return 1

def just_plot(prediction, record, temp_label, tempo):
    if((0 not in prediction) or (1 not in prediction) or (2 not in prediction) or (3 not in prediction) or (4 not in prediction)):
        return 0
    else:
        B1 = record[0 : prediction.index(1)]
        B1_loc = generate_loc(0, prediction.index(1))
    
        QRS = record[prediction.index(1) : prediction.index(2)]
        QRS_loc = generate_loc(prediction.index(1), prediction.index(2))
    
        B2 = record[prediction.index(2) : prediction.index(3)]
        B2_loc = generate_loc(prediction.index(2), prediction.index(3))
    
        T = record[prediction.index(3) : prediction.index(4)]
        T_loc = generate_loc(prediction.index(3), prediction.index(4))
        
        B3 = record[prediction.index(4) : len(prediction)]
        B3_loc = generate_loc(prediction.index(4), len(prediction))
    
        plt.plot(B1_loc,B1,color="red",label="B1")
        plt.plot(QRS_loc, QRS, color="green",label="QRS")
        plt.plot(B2_loc,B2,color="blue",label="B2")
        plt.plot(T_loc,T,color="orange",label="T")
        plt.plot(B3_loc,B3,color="black",label="B3")
        #plt.plot(tempo, color="maroon",label = "orig")
        #-----------------------------------------------------------
        #print(temp_label)
        B1t = tempo[0 : temp_label.index(1)]
        B1t_loc = generate_loc(0, temp_label.index(1))
    
        QRSt = tempo[temp_label.index(1) : temp_label.index(2)]
        QRSt_loc = generate_loc(temp_label.index(1), temp_label.index(2))
    
        B2t = tempo[temp_label.index(2) : temp_label.index(3)]
        B2t_loc = generate_loc(temp_label.index(2), temp_label.index(3))
    
        Tt = tempo[temp_label.index(3) : temp_label.index(4)]
        Tt_loc = generate_loc(temp_label.index(3), temp_label.index(4))
    
        B3t = tempo[temp_label.index(4) : len(temp_label)]
        B3t_loc = generate_loc(temp_label.index(4), len(temp_label))
    
        plt.plot(B1t_loc,B1t,color="red",label="B1t")
        plt.plot(QRSt_loc, QRSt, color="green",label="QRSt")
        plt.plot(B2t_loc,B2t,color="blue",label="B2t")
        plt.plot(Tt_loc,Tt,color="orange",label="Tt")
        plt.plot(B3t_loc,B3t,color="black",label="B3t")
    
        plt.title("5 states")
        plt.legend()
        plt.plot()
        plt.show()
        return 1


def truncated_labels(master_label):
    for index, item in enumerate(master_label):
        master_label[index] = item[0:800]
    return master_label


def compute_accuracy(curr_result, target_label):
    acc_count = 0
    for item1, item2 in zip(curr_result, target_label):
        if item1 == item2:
            acc_count = acc_count + 1
    return acc_count/len(curr_result) 


def convert_label(target_label):
    for ind, item in enumerate(target_label):
        if item == "B1" or item == "s0":
            target_label[ind] = 0
        elif item == "QRS" or item == "s1":
            target_label[ind] = 1
        elif item == "B2" or item == "s2":
            target_label[ind] = 2
        elif item == "T" or item == "s3":
            target_label[ind] = 3
        elif item == "B3" or item == "s4":
            target_label[ind] = 4
    return target_label


def test_model(trained_model, test_record, test_label):
    all_accuracies = [] 
    for record, target_label in zip(test_record, test_label): 
        curr_result = trained_model.predict(record, algorithm = "map")
        target_label = convert_label(target_label)
        curr_accuracy = compute_accuracy(curr_result, target_label)
        all_accuracies.append(curr_accuracy)
    return np.average(all_accuracies)

def compute_clusterwise_accuracy1(curr_result, target_label):
    if 0 not in curr_result:
        print("0 missed in prediction")
        return 0
    if 1 not in curr_result:
        print("1 missed in prediction")
        return 0
    if 2 not in curr_result:
        print("2 missed in prediction")
        return 0
    if 3 not in curr_result:
        print("3 missed in prediction")
        return 0
    if 4 not in curr_result:
        print("4 missed in prediction")
        return 0
    else:
        seg1 = [0, target_label.index(1)-1]
        seg2 = [target_label.index(1), target_label.index(2)-1]
        seg3 = [target_label.index(2), target_label.index(3)-1]
        seg4 = [target_label.index(3), target_label.index(4)-1]
        seg5 = [target_label.index(4), len(target_label)-1]
        all_seg = [seg1, seg2, seg3, seg4, seg5]
        cluster_acc = []
        for curr_seg in all_seg:
            acc = 0
            start, end = curr_seg[0], curr_seg[1]
            for index in range(curr_seg[0], curr_seg[1]+1):
                if curr_result[index] == target_label[index]:
                    acc = acc + 1
            cluster_acc.append(acc/(end-start+1))
        return cluster_acc


#cluster_accuracies = get_clusterwise_accuracy()
def get_clusterwise_accuracy(all_predictions, num_iterations, target_label):
    acc_b1 = []
    acc_qrs = []
    acc_b2 = []
    acc_t = []
    acc_b3 = []
    all_acc = [acc_b1, acc_qrs, acc_b2, acc_t, acc_b3]
    for i in range(0, num_iterations):
        acc_b1 = []
        acc_qrs = []
        acc_b2 = []
        acc_t = []
        acc_b3 = []
        all_acc = [acc_b1, acc_qrs, acc_b2, acc_t, acc_b3]
        curr_prediction_list = all_predictions[i]
        for curr_prediction, curr_target in zip(curr_prediction_list, target_label):
            curr_target_conv = convert_label(curr_target)
            res = compute_clusterwise_accuracy1(curr_prediction, curr_target_conv)
            if res != 0:
                for j in range(len(res)):
                    all_acc[j].append(res[j])
    return [np.average(all_acc[0]), np.average(all_acc[1]),\
            np.average(all_acc[2]), np.average(all_acc[3]),\
            np.average(all_acc[4])]
            
def compute_clusterwise_accuracy(curr_result, target_label):
    if 0 not in curr_result:
        print("0 missed in prediction")
        return 0
    if 1 not in curr_result:
        print("1 missed in prediction")
        return 0
    if 2 not in curr_result:
        print("2 missed in prediction")
        return 0
    if 3 not in curr_result:
        print("3 missed in prediction")
        return 0
    if 4 not in curr_result:
        print("4 missed in prediction")
        return 0
    else:
        ind1_pred = curr_result.index(1)
        ind2_pred = curr_result.index(2) - curr_result.index(1)
        ind3_pred = curr_result.index(3) - curr_result.index(2)
        ind4_pred = curr_result.index(4) - curr_result.index(3)
        ind5_pred = len(curr_result) - curr_result.index(4)
        ind6_target = target_label.index(1)
        ind7_target = target_label.index(2) - target_label.index(1)
        ind8_target = target_label.index(3) - target_label.index(2)
        ind9_target = target_label.index(4) - target_label.index(3)
        ind10_target = len(target_label) - target_label.index(4)
        avg_b1 = (abs(ind6_target - ind1_pred) / ind6_target) * 100
        avg_qrs = (abs(ind7_target - ind2_pred) / ind7_target) * 100 
        avg_b2 = (abs(ind8_target - ind3_pred) / ind8_target) * 100
        avg_t = (abs(ind9_target - ind4_pred) / ind9_target) * 100
        avg_b3 = (abs(ind10_target - ind5_pred) / ind10_target) * 100
        return [avg_b1, avg_qrs, avg_b2, avg_t, avg_b3]

    
def try_custom_model():
    trans_mat4 = \
    [
    [0.99649123, 0.00350877, 0, 0, 0, 0, 0],#B1
    [0, 0.99917079, 0.00282921, 0, 0, 0, 0],#QRS
    [0, 0, 0.80037254, 0.00562746, 0, 0, 0],#B2
    [0, 0, 0, 0.98715832, 0.01284168, 0, 0],#T
    #[0, 0, 0, 0.97015832, 0.01284168, 0, 0],#T
    [0, 0, 0, 0, 0.99999999, 0, 0],#B3
    [0, 1, 0, 0, 0, 0, 0],#start
    [0, 0, 0, 0, 0, 0, 0]#end
    ]
    dists4 = [
    NormalDistribution(-0.0010946812970090893,0.012122153012899626),#B1
        #NormalDistribution(0.019182441959117582, 0.14734029629923742),#QRS
        NormalDistribution(0.019182441959117582,  0.07500029629923742),#QRS
        #NormalDistribution(-0.000800980705388322, 0.04849948191505086),#B2
        NormalDistribution(-0.000800980705388322, 0.04849988891505086),#B2
        NormalDistribution(-0.002358891154442411, 0.048075165711373734),#T
        #NormalDistribution(-0.008238955732330999, 0.024333552338138366)#B3
        NormalDistribution(-0.008238955732330999,  0.021333552338138366)#B3
        ]
    trans_mat = \
    [
    [0.99649123, 0.00350877, 0, 0, 0, 0, 0],#B1
    [0, 0.99917079, 0.00132921, 0, 0, 0, 0],#QRS
    [0, 0, 0.80337254, 0.00562746, 0, 0, 0],#B2
    [0, 0, 0, 0.98715832, 0.01084168, 0, 0],#T
    #[0, 0, 0, 0.97015832, 0.01284168, 0, 0],#T
    [0, 0, 0, 0, 1, 0, 0],#B3
    [0, 1, 0, 0, 0, 0, 0],#start
    [0, 0, 0, 0, 0, 0, 1]#end
    ]
    dists = [
        NormalDistribution(-0.0010946812970090893,0.012122153012899626),#B1
        NormalDistribution(0.019182441959117582, 0.0900029629923742),#QRS
        NormalDistribution(-0.000800980705388322, 0.04749988891505086),#B2
        NormalDistribution(-0.002358891154442411, 0.04700105711373734),#T
        NormalDistribution(-0.008238955732330999, 0.015183552338138366)#B3
        ]
    
    starts = [1, 0, 0, 0, 0]
    ends = [0, 0, 0, 0, 1]
    
    model = HiddenMarkovModel.from_matrix(trans_mat4, dists4, starts, ends)
    model.bake()
    return model

def explore_average_lengths(trunc_master_label):
    len1 = []
    len2 = []
    len3 = []
    len4 = []
    len5 = []
    for list1 in trunc_master_label:
        len1.append(list1.index(1) - 0)
        len2.append(list1.index(2) - list1.index(1))
        len3.append(list1.index(3) - list1.index(2))
        len4.append(list1.index(4) - list1.index(3))
        len5.append(len(list1) - list1.index(4))
    print()
    print("Average legth of B1: ", np.average(len1))
    print("Average legth of QRS: ", np.average(len2))
    print("Average legth of B2: ", np.average(len3))
    print("Average legth of T: ", np.average(len4))
    print("Average legth of B3: ", np.average(len5))

def get_average_accuracy(all_predictions, num_iterations, target_label):
    all_accuracies = []
    for i in range(0, num_iterations):
        #print("iteration: ", i)
        curr_predictions = all_predictions[i]
        k = 0
        for k in range(0, len(target_label)):
            conv_target_label = convert_label(target_label[k])
            all_accuracies.append(compute_accuracy(curr_predictions[k], conv_target_label))
    return np.average(all_accuracies)
    
def shuffle_data(trans_master_record, backup_record, trunc_master_label, master_file_list):
    length = len(trans_master_record)
    x = [i for i in range(0, length)]
    np.random.seed()
    np.random.shuffle(x)
    new_trans_master_record = []
    new_backup_record = []
    new_trunc_master_label = []
    new_master_file_list = []
    for i in x:
        new_trans_master_record.append(copy(trans_master_record[i]))  
        new_backup_record.append(copy(backup_record[i]))
        new_trunc_master_label.append(copy(trunc_master_label[i]))
        new_master_file_list.append(copy(master_file_list[i]))
    return new_trans_master_record, new_backup_record, new_trunc_master_label,\
         new_master_file_list

def compute_dice_coefficient(all_test_predictions, num_iterations, test_label):
    print("computing dice")
    dice_b1, dice_qrs, dice_b2, dice_t, dice_b3 = ([] for i in range (5))
    all_dice = [dice_b1, dice_qrs, dice_b2, dice_t, dice_b3]
    for i in range(0, num_iterations):
        curr_prediction_list = all_test_predictions[i]
        for prediction, target in zip(curr_prediction_list, test_label):
            if((0 not in prediction) or (1 not in prediction) or (2 not in prediction) or (3 not in prediction) or (4 not in prediction)):
                #return 0
                continue
            else:
                conv_target_label = convert_label(target)
                seg_b1_pred = [0, prediction.index(1)-1]
                seg_b1_target = [0, conv_target_label.index(1)-1]
                seg_qrs_pred = [prediction.index(1), prediction.index(2)-1]
                seg_qrs_target = [conv_target_label.index(1), conv_target_label.index(2)-1]  
                seg_b2_pred = [prediction.index(2), prediction.index(3)-1]
                seg_b2_target = [conv_target_label.index(2), conv_target_label.index(3)-1]
                seg_t_pred = [prediction.index(3), prediction.index(4)-1]
                seg_t_target = [conv_target_label.index(3), conv_target_label.index(4)-1] 
                seg_b3_pred = [prediction.index(4), len(prediction)-1]
                seg_b3_target = [conv_target_label.index(4), len(conv_target_label)-1]
                pred_seg_list = [seg_b1_pred, seg_qrs_pred, seg_b2_pred, seg_t_pred, seg_b3_pred]
                tar_seg_list = [seg_b1_target, seg_qrs_target, seg_b2_target,  seg_t_target, seg_b3_target] 
                for i in range(0, 5):
                    x = generate_loc(pred_seg_list[i][0]+1, pred_seg_list[i][1]+1)
                    y = generate_loc(tar_seg_list[i][0]+1, tar_seg_list[i][1]+1)
                    intersection = 0
                    curr_dice = 0
                    for item1 in x:
                        if item1 in y:
                            intersection = intersection + 1
                    curr_dice = (2*intersection) / (len(x)+len(y))
                    all_dice[i].append(curr_dice)
    print()
    print("Dice similarity for B1: ", np.average(all_dice[0]))
    print("Dice similarity for QRS: ", np.average(all_dice[1]))
    print("Dice similarity for B2: ", np.average(all_dice[2]))
    print("Dice similarity for T: ", np.average(all_dice[3]))
    print("Dice similarity for B3: ", np.average(all_dice[4]))
            

def readfiles(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    master_record = []
    master_label = []
    master_file_index = {}
    master_file_list = []
    record_data_lengths = []
    try:
        os.remove("log.txt") 
    except OSError:
        pass
    log = open("log.txt", "w+")
    index = 0
    
    #removing files which are known to have missing t_peak, t_end or s_end 
    missing_label_files = \
    ['Subject_15File_7.csv', 'Subject_34File_1.csv',\
    'Subject_34File_2.csv', 'Subject_34File_3.csv', 'Subject_34File_4.csv',\
    'Subject_34File_7.csv', 'Subject_34File_8.csv', 'Subject_35File_2.csv',\
    'Subject_35File_3.csv', 'Subject_35File_8.csv', 'Subject_6File_1.csv',\
    'Subject_6File_4.csv', 'Subject_6File_8.csv', 'Subject_7File_4.csv', \
    'Subject_84File_7.csv', 'Subject_85File_4.csv','Subject_88File_7.csv',\
    'Subject_8File_2.csv', 'Subject_4File_1.csv'] 
    
    onlyfiles = list(set(onlyfiles) - set(missing_label_files)) 
    
    for file in onlyfiles:
        master_file_list.append(file)
        file_addr = mypath + file
        master_file_index[file] = index
        with open(file_addr,'r') as fp:
            read_file = fp.readlines()[6:]
        record_data_lengths.append(len(read_file))
        for i in range(0, len(read_file)):
            read_file[i] = read_file[i].rstrip()
        
        new_read_file = []
        for item in read_file:
            new_read_file.append(item.split(','))
        
        reading = []
        label = []
        
        for item in new_read_file:
            reading.append(float(item[0]))
            label.append(item[1])
            
        master_record.append(reading)
        master_label.append(label)
        
        index = index + 1
    
    #check and find if any readings have one or more labels missing
    print("read files...")   
    record_missing_labels, all_points = extract_points(master_record,\
                        master_label, master_file_list, log)
    print("returned record_missing_labels: ", len(record_missing_labels))
    
    master_label,diff = create_label_blocks(master_label)
    backup_record = copy(master_record)
    trans_master_record = transform(master_record)
    
    backup_label = copy(master_label)
    master_label = update_labels_to_digits( copy(master_label))
    
    trunc_master_label = truncated_labels(master_label)
    
    trans_master_record, backup_record, trunc_master_label, master_file_list = \
            shuffle_data(trans_master_record, backup_record, trunc_master_label, master_file_list)
    '''
    #85-15 split
    train_record = trans_master_record[0:104]
    train_label = trunc_master_label[0:104]
    train_files = master_file_list[0: 104]
    
    test_record = trans_master_record[104:122]
    test_label = trunc_master_label[104:122]
    test_files = master_file_list[104: 122]
    
    orig_test_record = backup_record[104:122]
    '''
    
    
    #65-35 split
    train_record = trans_master_record[0:80]
    train_label = trunc_master_label[0:80]
    train_files = master_file_list[0: 80]
    
    test_record = trans_master_record[80:122]
    test_label = trunc_master_label[80:122]
    test_files = master_file_list[80: 122]
    
    orig_test_record = backup_record[80:122]
    
    
    trained_model = trainModel(train_record, train_label, model=None)
    trained_model.bake()
    print("trained trans_mat : ")
    print(trained_model.dense_transition_matrix())
    
    print()
    print("model: ")
    print(trained_model)
    
    num_iterations = 1
    all_train_predictions = []
    for i in range(0, num_iterations):
        current_predictions = []
        for j in range(0, len(train_record)):
            current_predictions.append(trained_model.predict(train_record[j]))
        all_train_predictions.append(current_predictions)
    avg_train_accuracy = get_average_accuracy(all_train_predictions, num_iterations, train_label)
    print("Accuracy on train set is: ", avg_train_accuracy*100, "%")
    
    all_test_predictions = []
    for i in range(0, num_iterations):
        current_predictions = []
        for j in range(0, len(test_record)):
            current_predictions.append(trained_model.predict(test_record[j]))
        all_test_predictions.append(current_predictions)
    avg_test_accuracy = get_average_accuracy(all_test_predictions, num_iterations, test_label)
    print("Accuracy on test set is: ", avg_test_accuracy*100, "%")
    
    print()
    cluster_accuracies = get_clusterwise_accuracy(all_train_predictions, num_iterations, train_label)
    print("B1 train accuracy: ", cluster_accuracies[0]*100,"%") 
    print("QRS train accuracy: ", cluster_accuracies[1]*100,"%")
    print("B2 train accuracy: ", cluster_accuracies[2]*100,"%")
    print("T train accuracy: ", cluster_accuracies[3]*100,"%")
    print("B3 train accuracy: ", cluster_accuracies[4]*100,"%")
    print()
    
    #cluster_accuracies = get_clusterwise_accuracy(trained_model, test_record, test_label)
    cluster_accuracies = get_clusterwise_accuracy(all_test_predictions, num_iterations, test_label)
    print("B1 test accuracy: ", cluster_accuracies[0]*100,"%")
    print("QRS test accuracy: ", cluster_accuracies[1]*100,"%")
    print("B2 test accuracy: ", cluster_accuracies[2]*100,"%")
    print("T test accuracy: ", cluster_accuracies[3]*100,"%")
    print("B3 test accuracy: ", cluster_accuracies[4]*100,"%")
    
    compute_dice_coefficient(all_test_predictions, num_iterations, test_label)
    
    print("Plotting results...")
    ret = 1
    ret = new_plot1(all_test_predictions, test_record, orig_test_record,\
        test_label, test_files, num_iterations)
    if ret == 0:
        print("Plotting function returned 0 - one of the labels not in predictions...")
    else:
        print("Plotting completed successfully")
       
        
if __name__ == "__main__":
    filepath = "F:\\Courses\\IndStudy\\ActiveLearning-20180522T155005Z-001\\ActiveLearning\\ Sourabh Khasbag\\Final Submission\\ECGDATA\\ECGDATA\\Truncated\\Normative\\"
    readfiles(filepath)
