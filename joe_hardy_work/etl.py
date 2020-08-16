import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import numpy as np


def clean_file(file_path):
    c = list(map(lambda line: line.split(","), open(file_path, 'r').read().split("\n")))
    c1 = c[0]
    unique_names = [(ch, ch) for ch in c1 if ch != '']
    final_str = ''
    for i in range(len(c)):
        # print(i)
        row_info = []
        for j in range(len(c[1])):
            # print(j)
            if i == 0:
                if j % 3 != 2:  # or j==0:
                    # print(int(j/3),j % 3)
                    row_info.append(unique_names[int(j / 3.)][j % 3].strip() + " " + c[i + 1][j].strip())
            elif i == 1:
                continue
            else:
                if j % 3 != 2:
                    row_info.append(c[i][j])
            # print(row_info)
        final_str += ','.join(row_info) + "\n"
    return_string = StringIO(final_str)
    # print(return_string)
    return return_string


def get_timedelta(td):
    return td.total_seconds()


def align_timestamps(df):
    # print(df.columns)
    misaligned_rows = []

    num_offsets_changing = 0
    last_free_agents = 0
    # while num_offsets_changing < 1000:
    #   free_agents = {}
    col_changes = {}
    for i, row in df.iterrows():
        df_copy = df.copy()
        # print(row)
        try:
            timestamps = [datetime.strptime(row[k], "%m/%d/%y %H:%M") if not pd.isna(row[k]) else row[k] for k in
                          df.columns if "time" in k.lower()]
        except:
            timestamps = [row[k] for k in df.columns if "time" in k.lower()]
            # print(i,"iiii",timestamps)

        offsets = [max(d, timestamps[0]) - min(d, timestamps[0]) if not pd.isna(d) and not pd.isna(
            timestamps[0]) else timedelta(0) for d in timestamps]
        timestamp_cols_inx = [i for i, k in enumerate(df.columns) if "time" in k.lower()]
        timestamp_cols = dict(zip(timestamp_cols_inx, offsets))
        # print(timestamp_cols)
        for tc in timestamp_cols:

            if tc not in col_changes:
                col_changes[tc] = 0
            # print(tc, col_changes[tc],i)
            if get_timedelta(timestamp_cols[tc]) > 60 and (df.shape[0] - (i + col_changes[tc])) >= 0:
                df_copy.iloc[i + col_changes[tc]:, tc] = df_copy.iloc[i + col_changes[tc]:, tc].shift(1)
                col_changes[tc] += 1

        #     if max(map(lambda o: get_timedelta(o), offsets)) > 60:
        #         # print(i, timestamps,len(offsets), len(df.columns))
        #         # print(offsets)
        #         # print(max(map(lambda o: o.seconds, offsets)) > 60)
        #         for j in range(0, len(df.columns), 2):
        #             # print(j,int(j/2))
        #             add_to_free_agents = get_timedelta(offsets[int(j / 2)]) > 60
        #             if add_to_free_agents:
        #                 if j in free_agents:
        #                     #continue
        #                     free_agents[j].append(i)
        #                 else:
        #                     free_agents[j] = [i]
        #             # free_agents.extend([(i,j) for j in range(len(df.columns),2) if offsets[int(j/2)].seconds>60])
        # for e, j in enumerate(free_agents):
        #     for n, i in enumerate(sorted(free_agents[j])):
        #         #print(i,j,i + n + 1, i+n)
        #         rows_left = df.shape[0]-(i+n+1)
        #         if(rows_left<=0):
        #             break
        #         #print(df.shape[0],(i+n+1),rows_left)
        #         #print(e,free_agent,n,i,i+n)
        #         #print(df.iloc[i+n:,free_agent])
        #         df.iloc[i+n:,j]=df.iloc[i+n:,j].shift(1)
        #         break
        #         #df.iloc[i+n+1:,j] = df.iloc[i+n:,j]
        #         #print(df.iloc[i+n:,free_agent])
        #         #df.iloc[i + n + 1:, free_agent] = df.iloc[i + n:, free_agent]
        #         #df.iloc[i + n, j] = np.nan
        # # print(len(free_agents))
        # num_offsets_changing += 0 if len(free_agents) != last_free_agents else 1
        # last_free_agents = len(free_agents)

    return df_copy


machines = [
    "bridgeport1",
    "bridgeport2",
    "bridgeport3",
    "drillpress",
    "lathe"
]

dataset_paths = {
    machine: {'train1': "./data/{}week1-train.csv".format(machine),
              'train2': "./data/{}week2-train.csv".format(machine),
              'test': "./data/{}week3-test.csv".format(machine)}
    for machine in machines
}
datasets = {}
for machine in dataset_paths:
    datasets[machine] = {}
    for dataset_name in dataset_paths[machine]:
        cleaned_file_contents = clean_file(dataset_paths[machine][dataset_name])
        datasets[machine][dataset_name] = align_timestamps(pd.read_csv(cleaned_file_contents))
        # print(datasets[machine][dataset_name].describe())
        # print(datasets[machine][dataset_name].iloc[:-5,:6])
        df = datasets[machine][dataset_name]
        for i, row in df.iterrows():
            # print(row)
            try:
                timestamps = [datetime.strptime(row[k], "%m/%d/%y %H:%M") if not pd.isna(row[k]) else row[k] for k in
                              df.columns if "time" in k.lower()]
            except:
                timestamps = [row[k] for k in df.columns if "time" in k.lower()]
                # print(i,"iiii",timestamps)

            offsets = [max(d, timestamps[0]) - min(d, timestamps[0]) if not pd.isna(d) and not pd.isna(
                timestamps[0]) else timedelta(0) for d in timestamps]
            if max(map(lambda o: o.seconds, offsets)) > 60:
                print(i, machine, dataset_name, max(map(lambda o: o.seconds, offsets)) > 60)
                print(row)
                # break
