#
# Weather_Revenue_Network.py
# Making Predictions with the Neural Net between Weather and Profits (Fantastic Sams)
#
# Created by Vignesh Balasundaram on 05/01/2018
# Copyright (c) 2018 Vignesh Balasundaram. All rights reserved.

from bs4 import BeautifulSoup
import requests
import time
import csv
import calendar
import datetime
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SET YOUR START AND END DATE HERE:
# [month,day,year]
start_date = [1,1,2015]
end_date = [5,1,2018]

# DO YOU WANT VALUE OR NUMBER OF INSTANCES PER HOUR?
# R_or_P = 0 for Value
# R_or_P = 1 for Number of Instances
R_or_P = 1

# Train Neural Network Through Batching?
# Batching = 1 for yes
# Batching = 0 for no
Batching = 0

### SET NEURAL NETWORK INFORMATION HERE:
SIZE_OF_TEST_DATA = 500
EPOCHS = 10
HIDDEN_NODES = 5
LEARNING_RATE = 0.0000004
BATCH_SIZE = 8

# Already Have Weather Data? Set WDF = 1
WDF = 1
# Already Have Formatted Data File? Set FBD = 1
FBDF = 1

########################################################################################################################
########################################################################################################################

#GLOBAL VARIABLES

now = datetime.datetime.now()
s_day = start_date[1]
s_month = start_date[0]
s_year = start_date[2]

start_time = time.time()

e_day = end_date[1]
e_month = end_date[0]
e_year = end_date[2]

########################################################################################################################

### SCRAPE FOR WEATHER DATA IN GIVEN TIMESPAN

if WDF == 0:

    time_span = datetime.date(e_year,e_month,e_day) -\
                datetime.date(s_year,s_month,s_day)

    list_of_rows = [['Date','Time(CST)', 'hr', 'Temp.', 'Windchill', 'DewPoint', 'Humidity',
                     'Pressure', 'Visibility','WindDir','WindSpeed','GustSpeed',
                     'Precip','Events','Conditions']]

    day,month,year = s_day,s_month,s_year
    date = str(year) + "/" + str(month) + "/" + str(day)
    count = 0
    while year <= e_year:
        if year == e_year:
            m_end = e_month
        else:
            m_end = 12

        while month <= m_end:
            if year == e_year and month == e_month:
                d_end = e_day
            else:
                d_end = calendar.monthrange(year, month)[1]
            while day <= d_end:
                date = str(year) + "/" + str(month) + "/" + str(day)
                url = "https://www.wunderground.com/history/airport/KBNA/" + date +\
                      "/DailyHistory.html?req_city=&req_state=&req_statename=&reqdb."\
                      + "zip=&reqdb.magic=&reqdb.wmo="
                page = requests.get(url)
                psoup = BeautifulSoup(page.text, "html.parser")
                table = psoup.find("table", {"id": "obsTable"})
                p_l_row = list_of_rows[0]
                for row in table.findAll('tr'):
                    hc = 0
                    list_of_cells = [date]
                    for cell in row.findAll(["th", "td"]):
                        raw_text = cell.text
                        text = ''
                        for c in raw_text:
                            if c != "\n" and c != "\xa0" and c != ' ':
                                text += c
                        if hc == 1:
                            if list_of_cells[1][-2:] == 'AM':
                                new_cell = list_of_cells[1].split(":")[0]
                                if int(new_cell) == 12:
                                    new_cell = '0'
                                list_of_cells.append(new_cell)
                            else:
                                new_cell = int(list_of_cells[1].split(":")[0]) + 12
                                if new_cell == 24:
                                    new_cell = 12
                                list_of_cells.append(str(new_cell))
                            hc += 1
                            continue
                        if hc == 2:
                            if text == '-' or text == 'N/A%' or text == '\t\t-' or text == '\t':
                                if (p_l_row[hc+1]) != "Temp.":
                                    text = p_l_row[hc+1]
                                else:
                                    text = '40.0°F'
                        if hc == 3:
                            if text == '-' or text == 'N/A%' or text == '\t\t-' or text == '\t':
                                text = p_l_row[hc+1]
                        if hc == 6 or hc == 8 or hc == 12:
                            if text == '-' or text == 'N/A%' or text == '\t\t-' or text == '\t':
                                text = p_l_row[hc+2]
                        if hc == 10:
                            if text == 'N/A':
                                text = 'N/A'
                        if text == "Time(CST)" or text == "Time(CDT)":
                            break
                        else:
                            list_of_cells.append(text)

                        hc += 1

                    if list_of_cells != [date]:
                        if len(list_of_cells) == 15:
                            p_l_row = list_of_cells
                            list_of_rows.append(list_of_cells)
                        else:
                            new_list = list_of_cells[:4]
                            for i in range(15 - len(list_of_cells)):
                                new_list.append("-")
                            new_list += list_of_cells[4:]
                            p_l_row = new_list
                            list_of_rows.append(new_list)

                day+=1
                count+=1

                elapsed_time = float(time.time() - start_time)
                days_per_second = count / elapsed_time if elapsed_time > 0 else 0
                time_remaining = (time_span.days - count)/days_per_second
                sys.stdout.write(("\rProgress of Weather Scrape: " + \
                                  str(100*(int(count-1)/int(time_span.days)))[:4] +\
                                  "%") + "      Time Remaining: " +\
                                 str(int(time_remaining/60))[0:5] + " min")

            month+=1
            day = 1
        year+=1
        month =1

    data_filename = "Weather.csv"
    outfile = open(data_filename, "w", newline = '')
    writer = csv.writer(outfile)

    with open(data_filename, 'w', newline='') as t:
        writer = csv.writer(t)
        for row in list_of_rows:
            writer.writerow(row)


########################################################################################################################
########################################################################################################################

### FORMATTING THE BUSINESS DATA BY EACH HOUR OF OPERATION PER DAY
if FBDF == 0:
    Daily_Hours = {"6":[12,17], "0":[9,18], "1":[9,19], "2":[9,19],
                   "3":[9,19], "4":[9,18], "5":[8,17]}

    Holidays = ["1/1/2015","1/19/2015","2/16/2015","5/25/2015","7/3/2015","9/7/2015","10/12/2015","11/11/2015","11/26/2015","12/25/2015",
                "1/1/2016","1/18/2016","2/15/2016","5/30/2016","7/4/2016","9/5/2016","10/10/2016","11/11/2016","11/24/2016","12/26/2016",
                "1/2/2017","1/16/2017","2/20/2017","5/29/2017","7/4/2017","9/4/2017","10/9/2017","11/10/2017","11/23/2017","12/25/2017",
                "1/1/2018","1/15/2018","2/19/2018","5/28/2018","7/4/2018","9/3/2018","10/8/2018","11/12/2018","11/22/2018","12/25/2018"]

    day = s_day
    month = s_month
    year = s_year

    start_date = str(month) + "/" + str(day) + "/" + str(year)


    ### HELPER LIST TO FORMAT DAYS AND HOURS TO WORKING HOURS
    List_of_Rows = [['Date','Year','Month','Day','Hour', 'Weekday', 'Holiday', 'Temperature',
                     'Humidity', 'Visibility', 'WindSpeed','Precipitation','Condition','Profit']]

    while year <= e_year:

        if year == e_year:
            m_end = e_month
        else:
            m_end = 12

        while month <= m_end:
            if year == e_year and month == e_month:
                d_end = e_day
            else:
                d_end = calendar.monthrange(year,month)[1]

            while day <= d_end:
                date = str(month) + '/' + str(day) + '/' + str(year)
                holiday = 0
                for h in Holidays:
                    if date == h:
                        holiday = 1
                weekday = calendar.weekday(year,month,day)
                hours = Daily_Hours[str(weekday)]
                cur_time = hours[0]
                end_time = hours[1]
                if (int(year) < 2018 or int(day) < 20) and weekday == 6:
                    day += 1
                    continue

                while cur_time <= end_time:
                    hour_row = [date,str(year - s_year),str(month),str(day),str(cur_time),str(weekday),str(holiday),'','','','','','','']
                    List_of_Rows.append(hour_row)
                    cur_time += 1
                day += 1

            month += 1
            day = 1

        year += 1
        month = 1


    ########################################################################################################################

    conditions = {"HeavyRain":5, "HeavyIcePellets":5, "HeavySnow":5, "ThunderstormsandRain":5,
                  "HeavyThunderstormsandRain":5, "HeavyThunderstormswithSmallHail":5, "Rain":4,
                  "Snow":4, "SmallHail":4, "IcePellets":4, "LightIcePellets":4,
                  "LightThunderstormsandRain":4, "Thunderstorm":4, "Squalls":4, "Overcast":3,
                  "LightRain":3,"LightDrizzle":3,"LightSnow":3, "LightFreezingRain":3,
                  "LightFreezingDrizzle":3, "Fog":3, "Drizzle":3,"FunnelCloud":3,"MostlyCloudy":2,
                  "Haze":2, "PatchesofFog":2, "Mist":2, "LightFreezingFog":2, "Clear":1,
                  "ScatteredClouds":1, "PartlyCloudy":1, "Unknown":1}

    ### HELPER LIST TO HOLD ALL RAW WEATHER INFORMATION
    w_lists = [['Date','Year','Month','Day','Hour', 'Weekday', 'Holiday', 'Temperature',
                'Humidity', 'Visibility', 'WindSpeed','Precipitation','Condition','Profit']]

    max_temp = 0
    min_temp = 41
    max_vis = 0
    min_vis = 100
    max_precip = 0
    prev_row = []
    count = 0
    with open('Weather.csv') as csvfile:
        w = csv.reader(csvfile)
        prev = {"date":'0',"hour":'0'}
        for row in w:
            if row[0] == 'Date':
                continue

            date = row[0].split('/')
            date = date[1] + '/' + date[2] + '/' + date[0]
            hour = row[2]

            if date == prev["date"] and hour == prev["hour"]:
                continue

            d2 = date.split('/')
            month = d2[0]
            day = d2[1]
            year = d2[2]
            if (int(year) < 2018 and int(day) < 20) and calendar.weekday(int(year),int(month),int(day)) == 6:
                continue

            temperature_in_F = row[3][:-2]
            if float(temperature_in_F) > max_temp:
                max_temp = float(temperature_in_F)
            if float(temperature_in_F) < min_temp:
                min_temp = float(temperature_in_F)

            if row[6] == 'N/A%' or row[6] == '-':
                humidity = prev_row[8]
            else:
                humidity = float(row[6][:-1])/100

            if row[8] == '-':
                visibility = prev_row[9]
            else:
                visibility = row[8][:-2]

            if float(visibility) > max_vis:
                max_vis = float(visibility)
            if float(visibility) < min_vis:
                min_vis = float(visibility)

            if row[10] == '-':
                wspeed = prev_row[10]
            else:
                wspeed = row[10]

            if row[12] == "N/A":
                precipitation = "0"
            else:
                precipitation = row[12][:-2]
                if float(precipitation) > max_precip:
                    max_precip = float(precipitation)


            condition = row[14]
            hour_list = [date, year, month, day, hour, '', '', temperature_in_F, humidity, visibility, wspeed, precipitation, condition, '']
            prev_row = hour_list
            w_lists.append(hour_list)
            prev["date"] = date
            prev["hour"] = hour

    ########################################################################################################################

    ### HELPER LIST TO HOLD PROFIT DATA
    d_lists = [['Date', 'Hour','Profit']]

    max_profit = 0
    count = 0
    with open('Data.csv') as csvfile:
        d = csv.reader(csvfile)
        date = start_date
        day = s_day
        month = s_month
        year = s_year
        hours = Daily_Hours[str(calendar.weekday(year,month,day))]
        prev = {"date":start_date,"hour":hours[0]}
        profit = 0
        hour_list = [prev["date"],prev["hour"],'0']
        for row in d:
            if row[0] == 'dteday':
                continue
            date = row[0]
            hour = row[1]
            if date == prev["date"] and hour == prev["hour"]:
                if R_or_P == 1:
                    profit += 1
                if R_or_P == 0:
                    profit += float(row[2])
                hour_list[2] = str(profit)[:7]
            else:
                prev["date"] = date
                prev["hour"] = hour
                if profit > max_profit:
                    max_profit = profit
                d_lists.append(hour_list)
                if R_or_P == 1:
                    profit = 1
                if R_or_P == 0:
                    profit = float(row[2])
                hour_list = [date,hour,str(profit)[:7]]
        d_lists.append(hour_list)

    ########################################################################################################################


    ### WRITING A MASTER LIST TO HOLD ALL VARIABLES
    master_list = []
    for row in List_of_Rows:
        master_list.append(row)

    max_temp = (max_temp - 32) * 5.0/9.0
    min_temp = (min_temp - 32) * 5.0/9.0

    prev_row = []
    pcount = 0
    for m_row in master_list:
        if m_row[0] == "Date":
            continue
        date = m_row[0]
        hour = m_row[4]

        for row in w_lists:
            if row[0] == "Date":
                continue
            if row[0] == date and row[4] == hour:
                m_row[7] = str((((float(row[7]) - 32) * 5.0 / 9.0) - min_temp) / (max_temp - min_temp))[:7]
                m_row[8] = str(float(row[8])/100.00)
                m_row[9] = str((float(row[9]) - min_vis) / (max_vis - min_vis))[:7]
                if row[10] == 'Calm':
                    windspeed = "0"
                else:
                    ws = float(row[10][:-3])
                    if ws < 4:
                        windspeed = "1"
                    elif ws >= 4 and ws < 8:
                        windspeed = "2"
                    elif ws >= 8 and ws < 13:
                        windspeed = "3"
                    elif ws >= 13 and ws < 19:
                        windspeed = "4"
                    elif ws >= 19 and ws < 25:
                        windspeed = "5"
                    elif ws >= 25 and ws < 32:
                        windspeed = "6"
                    elif ws >= 32 and ws < 39:
                        windspeed = "7"
                    elif ws >= 39 and ws < 47:
                        windspeed = "8"
                    elif ws >= 47 and ws < 55:
                        windspeed = "9"
                    elif ws >= 55 and ws < 64:
                        windspeed = "10"
                    elif ws >= 64 and ws < 73:
                        windspeed = "11"
                    elif ws >= 73:
                        windspeed = "11"
                m_row[10] = windspeed
                m_row[11] = (str(float(row[11])/max_precip))[:7]
                m_row[12] = str(conditions[row[12]])
                m_row[13] = "0.0"
                break
        for row in d_lists:
            if row[0] == date and row[1] == hour:
                m_row[13] = str(float(row[2])/max_profit)[:7]
                break

        for i in range(len(m_row)):
            if m_row[i] == '':
                m_row[i] = prev_row[i]
        prev_row = m_row
        pcount+=1
        sys.stdout.write("\rProgress of Formatting Business Data: " + str(100*(int(pcount)/len(master_list)))[:4] + "%")
    ########################################################################################################################

    ### WRITING DATA TO FILE
    if R_or_P == 0:
        data_filename = "Revenue_by_Hour_Data_T.csv"
    else:
        data_filename = "NumPeople_per_Hour_Data_T.csv"

    outfile = open(data_filename, "w", newline = '')
    writer = csv.writer(outfile)

    with open(data_filename, 'w', newline='') as t:
        writer = csv.writer(t)
        for row in master_list:
            writer.writerow(row)
            pcount +=1

########################################################################################################################
########################################################################################################################

### Making Predictions with the Neural Network for Data [Fantastic Sams]

# Encapsulate our neural network in a class
class NeuralNetwork:
    def __init__(self, data, targets, hidden_nodes = 35, output_nodes = 1, learning_rate = 0.3):
        np.random.seed(1)

        self.pre_process_data(data, targets)

        self.init_network(len(self.data), hidden_nodes, output_nodes, learning_rate)

    def pre_process_data(self, d, t):
        self.data = d
        self.revenue = t

        self.data_size = len(self.data)
        self.revenue_size = len(self.revenue)


    def init_network(self, input_nodes,hidden_nodes,output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0,self.output_nodes**-0.5,(self.hidden_nodes,self.output_nodes))

        self.layer_0 = np.zeros((1,self.input_nodes))

    def update_input_layer(self,dataset):

        self.layer_0 *= 0
        for data in dataset:
            self.layer_0[0] += data

    def sigmoid(self,x):
        return (1.0 / (1.0 + np.exp(-x)))

    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)

    def relu(self,x):
        if x > 0:
            return x[0]
        else:
            return 0

    def relu_output_2_derivative(self,x):
        if x > 0 :
            return 1
        else:
            return 0

    def train(self, data, targets):
        tcount = 0 # for debugging purposes
        assert(len(data) == len(targets))

        correct_so_far = 0
        start = time.time()

        self.delta_weights_0_1 = np.zeros(self.weights_0_1.shape)  # num_inputs x 10 matrix
        self.delta_weights_1_2 = np.zeros(self.weights_1_2.shape)  # 10 x 1 matrix

        for i in range(len(data)):

            dataset = data[i]

            y = float(targets[i][0])

            self.update_input_layer(dataset)

            layer_1_input = np.dot(self.layer_0, self.weights_0_1)
            layer_1_output = self.sigmoid(layer_1_input)

            layer_2_input = np.dot(layer_1_output, self.weights_1_2)
            layer_2_output = self.relu(layer_2_input[0])

            layer_2_error = y - layer_2_output
            layer_2_error_term = layer_2_error * self.relu_output_2_derivative(layer_2_output)
            layer_1_error = layer_2_error_term * self.weights_1_2
            layer_1_error_terms = layer_1_error * self.sigmoid_output_2_derivative(self.sigmoid(layer_1_output)).T

            self.delta_weights_1_2 += np.dot(layer_2_error_term, layer_1_output).T
            self.delta_weights_0_1 += np.dot(layer_1_error_terms, self.layer_0).T

            self.weights_1_2 += self.delta_weights_1_2 * (self.learning_rate)
            self.weights_0_1 += self.delta_weights_0_1 * (self.learning_rate)

            pred = np.abs(layer_2_error)
            if pred < (2.0/15):
                correct_so_far += 1

            Training_Accuracy = correct_so_far * 100 / float(i+1)
            tcount += 1 # update test

            if Batching == 1:
                continue

            elapsed_time = float(time.time() - start)
            rows_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rNeural Network Train Progress: " + str(100 * i / float(len(data)))[:4] \
                             + "% Speed(rows/sec): " + str(rows_per_second)[0:5] \
                             + " #Correct: " + str(correct_so_far) + " #Trained: " + str(i + 1)\
                             + " Training Accuracy: " + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if (i%25000 == 0):
                print("")

        return Training_Accuracy

    def test(self, testing_rows, testing_revenues):
        correct = 0

        start = time.time()

        for i in range(len(testing_rows)):

            data = self.run(testing_rows[i])
            target = float(testing_revenues[i][0])
            pred = np.abs(target - data)
            if pred < (2.0/15):
                correct += 1

            elapsed_time = float(time.time() - start)
            rows_per_second = 1/elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rNeural Network Test Progress: " + str(100 * i / float(len(testing_rows)))[:4] \
                             + "% Speed(rows/sec): " + str(rows_per_second)[0:5] \
                             + " #Correct: " + str(correct) + " #Trained: " + str(i + 1)\
                             + " Testing Accuracy: " + str(correct * 100 / float(i+1))[:4] + "%")

    def run(self, feature):

        self.update_input_layer(feature)

        layer_1_input = np.dot(self.layer_0, self.weights_0_1)
        layer_1_output = self.sigmoid(layer_1_input)

        layer_2_input = np.dot(layer_1_output, self.weights_1_2)
        layer_2_output = self.relu(layer_2_input[0])

        return layer_2_output

########################################################################################################################


### ORGANIZE FILE INFORMATION
if R_or_P == 0:
    csv_file_name = "Revenue_by_Hour_Data_T.csv"
else:
    csv_file_name = "NumPeople_per_Hour_Data_T.csv"

with open(csv_file_name) as csvfile:
    d = csv.reader(csvfile)
    rev = []
    for row in d:
        r = [row[-1]]
        rev.append(r)
revenues = rev[1:]

d = pd.read_csv(csv_file_name)
dummy_fields = ['Year', 'Month', 'Day', 'Hour', 'Weekday', 'WindSpeed', 'Condition']
for each in dummy_fields:
    dummies = pd.get_dummies(d[each], prefix=each, drop_first=False)
    d = pd.concat([d, dummies],axis=1)

targets = d['Profit']

fields_to_drop = ['Date','Year','Month','Day','Hour','Weekday','WindSpeed','Condition','Profit']
data = d.drop(fields_to_drop,axis=1)

max = 0
count = 0 # for debugging purposes
for i in revenues:
    if i[0] == '':
        i[0] = '0.0'
    if float(i[0]) > max:
        max = float(i[0])

for i in revenues:
    i[0] = str(float(i[0])/max)

train_data = np.array(data[:-SIZE_OF_TEST_DATA])
train_targets = revenues[:-SIZE_OF_TEST_DATA]
test_data = np.array(data[-SIZE_OF_TEST_DATA:])
test_targets = revenues[-SIZE_OF_TEST_DATA:]

### SET NETWORK INFORMATION
iterations = EPOCHS
hidden_n = HIDDEN_NODES
learning_r = LEARNING_RATE
batch_size = BATCH_SIZE

### CREATE AND TRAIN NETWORK
FSNN = NeuralNetwork(train_data, train_targets, hidden_nodes=hidden_n, learning_rate=learning_r)

if Batching == 0:
    FSNN.train(train_data,train_targets)
    print('\n')
    FSNN.test(test_data,test_targets)
    exit()

########################################################################################################################
### BATCHING TESTS BELOW
########################################################################################################################

def MSE(y, Y):
    sum = 0
    for item in range(len(y)):
        sum += (y[item][0] - float(Y[item][0]))**2
    return sum

losses = {'train':[],'validation':[]}
for i in range(iterations):

    features = [None] * batch_size
    targets = [None] * batch_size

    randlist = np.random.randint(0,9939,batch_size)

    for l in range(len(randlist)):
        index = randlist[l]
        features[l] = train_data[index]
        targets[l] = train_targets[index]
    features = np.array(features)
    FSNN.train(features,targets)

    tr_d = []
    te_d = []
    for feature in train_data:
        tr_d.append([FSNN.run(feature)])
    for feature in test_data:
        te_d.append([FSNN.run(feature)])

    train_loss = MSE(tr_d,train_targets)
    val_loss = MSE(te_d,test_targets)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * i/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:7] \
                     + " ... Validation loss: " + str(val_loss)[:7])
    sys.stdout.flush()
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.show()

FSNN.test(test_data,test_targets)

exit()