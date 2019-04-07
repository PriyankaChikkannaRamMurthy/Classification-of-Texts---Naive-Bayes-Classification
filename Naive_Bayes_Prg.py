import os
import random
import re
import numpy as np

from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
nonuse_words = ['newsgroups', 'xref', 'path', 'from', 'subject', 'sender', 'organisation', 'apr','gmt',
               'last','better','never','every','even','two','good','used','first','need','going','must',
               'really','might','well','without','made','give','look','try','far','less','seem','new','make',
               'many','way','since','using','take','help','thanks','send','free','may','see','much','want','find',
               'would','one','like','get','use','also','could','say','us','go','please','said','set','got','sure',
               'come','lot','seems','able','anything','put', '--', '|>', '>>', '93', 'xref', 'cantaloupe.srv.cs.cmu.edu',
               '20', '16', "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'", '21', '19', '10', '17', '24',
               'reply-to:', 'thu', 'nntp-posting-host:', 're:','25''18'"i'd"'>i''22''fri,''23''>the','references:','xref:',
               'sender:','writes:','1993','organization:','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd','4th', '5th', '6th', '7th', '8th', '9th', '10th']
training_words = []
testing_words = []
word_dict_train = {}
word_dict_test = {}
training_newdict = {}
testing_newdict = {}
final_dict = {}
file_no = 1
X_train = []
new_list = []
frequency = []
u_word = []
count = []
word_train = []
word_test = []
class_train = []
class_test = []

file_names = list()
dir_name = '20_newsgroups/'
folder_list = os.listdir(dir_name)


for each_folder in folder_list:
    folder_path = os.path.join(dir_name, each_folder)
    #print(folder_path)
    #print(each_folder)
    file_list = os.listdir(folder_path)
    random.shuffle(file_list)
    half_list = len(file_list) // 2
    first_half_names= file_list[:half_list]
    #print(first_half_names)
    second_half_names = file_list[half_list:]
    for x in first_half_names:
        training_newdict[x] = each_folder
        train_x = []
        class_train.append(each_folder)
        path = dir_name + each_folder + '/' + x
        #print(path)
        #opening each file to be read
        #removing metadata from each file
        with open(path, 'r') as file_text_lines:
            for each_line in file_text_lines:
                if 'Lines:' in each_line:
                    #print("foundddddddddd")
                    for each_line in file_text_lines:
                        for word in re.findall(r"[A-Za-z]+", each_line):
                            if not word.lower() in stop_words:
                                if not word.lower() in nonuse_words:
                                    if word_dict_train.get(word.lower()) != None:
                                        word_dict_train[word.lower()] += 1
                                        train_x.append(word.lower())
                                    else:
                                        word_dict_train[word.lower()] = 1
                                        train_x.append(word.lower())
        training_words.append(train_x)

    for y in second_half_names:
        testing_newdict[y] = each_folder
        test_x = []
        class_test.append(each_folder)
        path = dir_name + each_folder + '/' + y
        # print(path)
        # opening each file to be read
        # removing metadata from each file
        with open(path, 'r') as file_text_lines:
            for each_line in file_text_lines:
                if 'Lines:' in each_line:
                    # print("foundddddddddd")
                    for each_line in file_text_lines:
                        for word in re.findall(r"[A-Za-z]+", each_line):
                            if not word.lower() in stop_words:
                                if not word.lower() in nonuse_words:
                                    if word_dict_test.get(word.lower()) != None:
                                        word_dict_test[word.lower()] += 1
                                        test_x.append(word.lower())
                                    else:
                                        word_dict_test[word.lower()] = 1
                                        test_x.append(word.lower())
        testing_words.append(test_x)



#train_list1=set(training_words)
#print(len(train_dict1))
#unique_train_keys = list(word_dict_train.keys())
#unique_train_values = list(word_dict_train.values())
#print(len(unique_train))
#test_list1=set(training_words)
#print(len(test_dict1))
#unique_test = list(word_dict_test.keys())
#print(len(unique_test))
#print(type(word_dict_train))
#sorted(word_dict_train.items(), key=lambda x: x[1])

list_train_words = list(word_dict_train.keys())
unique_train_sort = sorted(word_dict_train.items(), key= operator.itemgetter(1), reverse= True)
new_train_sort = unique_train_sort[0:5000]
for tr_word in training_words:
    arr =[]
    for each_word in new_train_sort:
        if(each_word in tr_word):
            arr.append(tr_word.count(each_word))
        else:
            arr.append(0)
    word_train.append(arr)
np_word_train = np.asarray(word_train)
print(np_word_train.shape)
np_class_train = np.asarray(class_train)
print(np_class_train.shape)
#print(type(word_dict_train))
#unique_train_keys = list(word_dict_train.keys())
#unique_train_values = list(word_dict_train.values())
list_test_words = list(word_dict_test.keys())
unique_test_sort = sorted(word_dict_test.items(), key= operator.itemgetter(1), reverse= True)
new_test_sort = unique_test_sort[0:5000]
for te_word in testing_words:
    arr2 =[]
    for each_word1 in new_test_sort:
        if(each_word1 in te_word):
            arr2.append(te_word.count(each_word1))
        else:
            arr2.append(0)
    word_test.append(arr2)
np_word_test = np.asarray(word_test)
print(np_word_test.shape)
np_class_test = np.asarray(class_test)
print(np_class_test.shape)

def model_fit(np_word_train, np_class_train):
    res_dict = {}
    for each_label in folder_list:
        res_dict[each_label] = {}
        res_dict["TOTAL_DATA"] = len(np_class_train)
        np_word_train_cur = np_word_train[np_class_train == each_label]
        all_class, c = np.unique(np_class_train, return_counts=True)
    for m in range(len(all_class)):
        for n in range(5000):
            res_dict[each_label][new_train_sort[n]] = np_word_train_cur[:,n].sum()
        res_dict[each_label]["TOTAL_COUNT"]=c[m]
    return res_dict

def probability_log(res_dict, word, folder):
    result_value = np.log(res_dict[folder]["TOTAL_COUNT"]) - np.log(res_dict["TOTAL_DATA"])
    for num in range(len(word)):
        if (word[num]  in res_dict[folder].keys()):
            cur_file = word[num]
            cur_file_count = res_dict[folder][cur_file] + 1
            res_dict_count = res_dict[folder]["TOTAL_COUNT"] + len(res_dict[folder].keys())
            cur_probability = np.log(cur_file_count) - np.log(res_dict_count)
            result_value += cur_probability
        else:
            continue
    return result_value

def ind_probability(word, res_dict):
    base_val = -1
    prob_val = -1000
    keys_dict = res_dict.keys()
    for folder in keys_dict :
        if (folder == "TOTAL_DATA"):
            continue
        directory_prob = probability_log(res_dict, word, folder)
        if (directory_prob > prob_val):
            prob_val = directory_prob
            base_val = folder
    return base_val


def predictions(np_word_test, res_dict):
    newly_predicted_class = []
    for k in range(len(np_word_test)):
        predicted_class = ind_probability(np_word_test[k, :], res_dict)
        newly_predicted_class.append(predicted_class)
    return newly_predicted_class

def new_possibility(res_dict, np_word_test):
    class_possibility = []
    for word in np_word_test:
        pred_class = ind_probability(res_dict, word)
        class_possibility.append(pred_class)
    return class_possibility

res_dict = model_fit(np_word_train, np_class_train)
print(type(res_dict))

Bayes_output = new_possibility(res_dict, np_word_test)
Bayes_output = np.asarray(Bayes_output)
#print(Bayes_output.shape)
newly_predicted_class = predictions(np_word_test, Bayes_output)
print("Accuracy for Naive Bayes found to be : ")
print(accuracy_score(np_class_test, newly_predicted_class))
print("Classification report for Naive Bayes found to be : ")
print(classification_report(np_class_test, newly_predicted_class))
