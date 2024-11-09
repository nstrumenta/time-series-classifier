import json
from mcap.reader import make_reader
import statistics
import pandas as pd
import datetime

def add_labels(root_dir, dataframe, time_series_start, median_time_offset, labels_file_name):
    file_name = root_dir + '/data/' + labels_file_name + '.labels.json' #--------- Only Bamberg currently ---------
    dataframe["labels"] = "NO_LABEL"
    with open(file_name, "rb") as f:
        json_data = json.load(f)
        events = json_data["events"]
        for json_object in events:
            label = json_object["metadata"]
            if len(label) != 0:
                startTime = json_object["startTime"]
                endTime = json_object["endTime"]

                # fullStartTimeTs = str(startTime["sec"] + (startTime["nsec"] * 1000000000))
                # fullEndTimeTs = str(endTime["sec"] + (endTime["nsec"] * 100000000))

                fullStartTimeTs = str(startTime["sec"]) + str(startTime["nsec"]) # multiply by 1000000000
                fullEndTimeTs = str(endTime["sec"]) + str(endTime["nsec"])

                startTimeTsFormatted = fullStartTimeTs.ljust(19, '0')
                endTimeTsFormatted = fullEndTimeTs.ljust(19, '0')

                startTimeTd = pd.to_timedelta(startTimeTsFormatted) - pd.to_timedelta(time_series_start)
                endTimeTd = pd.to_timedelta(endTimeTsFormatted) - pd.to_timedelta(time_series_start)

                if startTimeTd > datetime.timedelta(days=0) and endTimeTd > datetime.timedelta(days=0) and list(label.keys())[0] == "voltage_level":
                    startTimeTd = startTimeTd.round(f'{median_time_offset}N')
                    endTimeTd = endTimeTd.round(f'{median_time_offset}N')
                    dataframe.loc[startTimeTd:endTimeTd, "labels"] = str(list(label.values())[0])

    return dataframe

def read_dataset(root_dir, dataset_name, only_pd=False, create_debug_file=False):
    file_name = root_dir + '/data/' + dataset_name + '.mcap' #-------Only Bamberg currently---------
    topics = ["PEEK_RAW", "GYRO_RAW", "PEEK_PRESSURE", "ACCEL_RAW", "AR", "TIMESTAMP_FULL", "START_VIDEO"]
    datasets_dict = {}
    for topic in topics:
        datasets_dict[topic] = {}
    with open(file_name, "rb") as f:
        reader = make_reader(f)
        i = 0
        for schema, channel, message in reader.iter_messages(): #define specific topics using reader.iter_messages(topics=["/{{topic}}"]) ex. /diagnostics
            json_str = message.data.decode('utf8').replace("'", '"')
            json_data = json.loads(json_str)
            if json_data.get("correctedTs") != None:
                datasets_dict[channel.topic][str(json_data["correctedTs"])] = json_data["values"]   #"ts", "appTs" or "correctedTs"
            else:
                datasets_dict[channel.topic][str(json_data["ts"])] = json_data["values"]            #"ts", "appTs" or "correctedTs"

    #Calculate time diff vector
    sortedTimes = sorted(datasets_dict['PEEK_RAW'].keys())
    n_timestamps = len(sortedTimes)
    timeDiffVec = [abs(int(sortedTimes[i]) - int(sortedTimes[i+1])) for i in range(n_timestamps-1)]
    median_time_offset = statistics.median(sorted(timeDiffVec)) # Answer in ns
    resample_frequency = 1000000000/median_time_offset # Answer in Hz
    print(f'Resampling at {resample_frequency} Hz')
    #divide all the time series further into separate components
    subtopics = ["PEEK_RAW_E_1", "PEEK_RAW_E_2", "PEEK_RAW_E_3", "PEEK_RAW_M_X", "PEEK_RAW_M_Y", "PEEK_RAW_M_Z", "GYRO_RAW_X", "GYRO_RAW_Y", "GYRO_RAW_Z", "ACCEL_RAW_X", "ACCEL_RAW_Y", "ACCEL_RAW_Z", "PEEK_PRESSURE"] #, "AR", "TIMESTAMP_FULL", "START_VIDEO"]
    datasets_dict_grouping = {}
    prev_real_peek_values = [0,0,0,0,0,0]
    prev_real_gyro_values = [0,0,0]
    prev_real_accel_values = [0,0,0]
    prev_real_press_values = [0]
    for subtopic in subtopics:
        datasets_dict_grouping[subtopic] = {}
    for ts in list(datasets_dict["PEEK_RAW"]):
        for i in range(6):
            datasets_dict_grouping[f"{subtopics[i]}"][ts] = list(datasets_dict["PEEK_RAW"][ts])[i]
    for ts in list(datasets_dict["GYRO_RAW"]):
        for i in range(3):
            datasets_dict_grouping[f"{subtopics[i+6]}"][ts] = list(datasets_dict["GYRO_RAW"][ts])[i]
    for ts in list(datasets_dict["ACCEL_RAW"]):
        for i in range(3):
            datasets_dict_grouping[f"{subtopics[i+9]}"][ts] = list(datasets_dict["ACCEL_RAW"][ts])[i]
    for ts in list(datasets_dict["PEEK_PRESSURE"]):
        datasets_dict_grouping[f"{subtopics[12]}"][ts] = list(datasets_dict["PEEK_PRESSURE"][ts])
    # for ts in list(datasets_dict["PEEK_RAW"]):
    #     for i in range(6):
    #         if not np.isnan(list(datasets_dict["PEEK_RAW"][ts])[0]):
    #             temp = list(datasets_dict["PEEK_RAW"][ts])[i]
    #         else:
    #             temp = prev_real_peek_values[i]
    #         datasets_dict_grouping[f"{subtopics[i]}"][ts] = temp
    #         prev_real_peek_values[i] = temp
    # for ts in list(datasets_dict["GYRO_RAW"]):
    #     for i in range(3):
    #         if not np.isnan(list(datasets_dict["GYRO_RAW"][ts])[0]):
    #             temp = list(datasets_dict["GYRO_RAW"][ts])[i]
    #         else:
    #             temp = prev_real_gyro_values[i]
    #         datasets_dict_grouping[f"{subtopics[i+6]}"][ts] = temp
    #         prev_real_gyro_values[i] = temp
    # for ts in list(datasets_dict["ACCEL_RAW"]):
    #     for i in range(3):
    #         if not np.isnan(list(datasets_dict["ACCEL_RAW"][ts])[0]):
    #             temp = list(datasets_dict["ACCEL_RAW"][ts])[i]
    #         else:
    #             temp = prev_real_accel_values[i]
    #         datasets_dict_grouping[f"{subtopics[i+9]}"][ts] = temp
    #         prev_real_accel_values[i] = temp
    # for ts in list(datasets_dict["PEEK_PRESSURE"]):
    #     if not np.isnan(list(datasets_dict["PEEK_PRESSURE"][ts])[0]):
    #         temp = list(datasets_dict["PEEK_PRESSURE"][ts])
    #     else:
    #         temp = prev_real_press_values
    #     datasets_dict_grouping[f"{subtopics[12]}"][ts] = temp
    #     prev_real_press_values = temp

    datasets_dict_grouping["AR"] = datasets_dict["AR"]
    datasets_dict_grouping["TIMESTAMP_FULL"] = datasets_dict["TIMESTAMP_FULL"]
    datasets_dict_grouping["START_VIDEO"] = datasets_dict["START_VIDEO"]
    df_PEEK_RAW_E_1 =   pd.DataFrame.from_dict(datasets_dict_grouping["PEEK_RAW_E_1"], orient="index")
    df_PEEK_RAW_E_2 =   pd.DataFrame.from_dict(datasets_dict_grouping["PEEK_RAW_E_2"], orient="index")
    df_PEEK_RAW_E_3 =   pd.DataFrame.from_dict(datasets_dict_grouping["PEEK_RAW_E_3"], orient="index")
    df_PEEK_RAW_M_X =   pd.DataFrame.from_dict(datasets_dict_grouping["PEEK_RAW_M_X"], orient="index")
    df_PEEK_RAW_M_Y =   pd.DataFrame.from_dict(datasets_dict_grouping["PEEK_RAW_M_Y"], orient="index")
    df_PEEK_RAW_M_Z =   pd.DataFrame.from_dict(datasets_dict_grouping["PEEK_RAW_M_Z"], orient="index")
    df_GYRO_RAW_X =     pd.DataFrame.from_dict(datasets_dict_grouping["GYRO_RAW_X"], orient="index")
    df_GYRO_RAW_Y =     pd.DataFrame.from_dict(datasets_dict_grouping["GYRO_RAW_Y"], orient="index")
    df_GYRO_RAW_Z =     pd.DataFrame.from_dict(datasets_dict_grouping["GYRO_RAW_Z"], orient="index")
    df_ACCEL_RAW_X =    pd.DataFrame.from_dict(datasets_dict_grouping["ACCEL_RAW_X"], orient="index")
    df_ACCEL_RAW_Y =    pd.DataFrame.from_dict(datasets_dict_grouping["ACCEL_RAW_Y"], orient="index")
    df_ACCEL_RAW_Z =    pd.DataFrame.from_dict(datasets_dict_grouping["ACCEL_RAW_Z"], orient="index")
    df_PEEK_PRESSURE =  pd.DataFrame.from_dict(datasets_dict_grouping["PEEK_PRESSURE"], orient="index")
    df_array = [df_PEEK_RAW_E_1, df_PEEK_RAW_E_2, df_PEEK_RAW_E_3, df_PEEK_RAW_M_X, df_PEEK_RAW_M_Y, df_PEEK_RAW_M_Z, \
                df_GYRO_RAW_X, df_GYRO_RAW_Y, df_GYRO_RAW_Z, df_ACCEL_RAW_X, df_ACCEL_RAW_Y, df_ACCEL_RAW_Z, df_PEEK_PRESSURE]

    # super_time_stamp = list(datasets_dict_grouping["TIMESTAMP_FULL"].keys())[0] #Super time stamp from TIMESTAMP_FULL doesn't seem to work
    super_time_stamp = df_array[0].index[0] # Super time stamp from E_1 first value
    resampled_array = []
    df_resampled_array = pd.DataFrame()
    for df in df_array:
        df.index = pd.to_timedelta(df.index)
        df = df.sort_index()
        time_series_start = df.index[0]
        df.index = df.index.map(lambda x: x - df.index[0])
        pre_resampled = df
        resampled = df.resample(f'{median_time_offset}N').ffill()
        resampled_array.append(resampled)


    df_resampled_array = pd.concat(resampled_array, axis=1)
    df_resampled_array.columns = subtopics

    # Label data
    labeled_df = add_labels(root_dir, df_resampled_array, super_time_stamp, median_time_offset, labels_file_name=dataset_name)

    # Trim away completely empty vectors
    labeled_df_trimmed = labeled_df.dropna(subset=subtopics)

    if only_pd:
        return labeled_df_trimmed

    if create_debug_file:
        labeled_df.to_csv(root_dir + '/output/' + dataset_name + "_before_trimming.csv")
        labeled_df_trimmed.to_csv(root_dir + '/output/' + dataset_name + ".csv")

    if shuffle_dataset and not use_rolling_window:
        Xy_train, Xy_test = train_test_split(labeled_df_trimmed, test_size=0.2, shuffle=True)
        if create_debug_file:
            Xy_train.to_csv(root_dir + '/output/' + dataset_name + "_shuffled_training.csv")
    else:
        Xy_train, Xy_test = train_test_split(labeled_df_trimmed, test_size=0.2, shuffle=False)

    X_train = Xy_train.loc[:, Xy_train.columns != 'labels']
    y_train = Xy_train["labels"]
    X_test = Xy_test.loc[:, Xy_train.columns != 'labels']
    y_test = Xy_test["labels"]
    X_train = np.nan_to_num(X_train.to_numpy())
    y_train = y_train.to_numpy()
    X_test = np.nan_to_num(X_test.to_numpy())
    y_test = y_test.to_numpy()
    #Normalize data:
    min_max_scaler = MinMaxScaler()
    X_train_test = np.concatenate((X_train, X_test), axis=0)
    min_max_scaler.fit(X_train_test)
    # min_max_scaler.get_params()
    X_train_minmax = min_max_scaler.transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)

    datasets_dict[dataset_name] = (X_train_minmax.copy(), y_train.copy(), X_test_minmax.copy(), y_test.copy())

    return datasets_dict
# example_dataset = read_dataset('/content/drive/MyDrive/Master', "Sensor_Log_2023-04-11_11_56_33", 100)
# # for i, x in enumerate(example_dataset["Sensor_Log_2023-04-11_11_56_33.mcap"][0]):
# #     print(f"{i}: {x}")
# print("X_train: ", example_dataset["Sensor_Log_2023-04-11_11_56_33"][0][:3])
# print("y_train: ", example_dataset["Sensor_Log_2023-04-11_11_56_33"][1][:3])
# print("X_test: ", example_dataset["Sensor_Log_2023-04-11_11_56_33"][2][:3])
# print("y_test: ", example_dataset["Sensor_Log_2023-04-11_11_56_33"][3][:3])