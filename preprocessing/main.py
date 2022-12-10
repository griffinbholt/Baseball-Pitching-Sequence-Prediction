from pybaseball import statcast
from pybaseball import cache
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, average_precision_score, \
    roc_auc_score

START_DATE = '2010-04-04'
END_DATE = '2019-10-30'
STATCAST_ORIGINAL_LOCATION_DATA = './statcast_dataset.csv'
MIN_YEAR = 2010
# SI and FT synonymous. No data has FT.
PITCHES = {"CH", "CU", "EP", "FC", "FF", "FO", "FS", "KC", "KN", "PO", "SC", "SI", "SL"}
PITCHES_LIST = ["CH", "CU", "EP", "FC", "FF", "FO", "FS", "KC", "KN", "PO", "SC", "SI", "SL"]
VERLANDER_PITCHES_LIST = ["CH", "CU", "FF", "PO", "SI", "SL"]
PITCHES_PERCENTAGES = ["percent_ch", "percent_cu", "percent_ep", "percent_fc", "percent_ff", "percent_fo", "percent_fs",
                       "percent_kc", "percent_kn", "percent_po", "percent_sc", "percent_si", "percent_sl"]
PITCHES_TENDENCY = ["tendency_ch", "tendency_cu", "tendency_ff", "tendency_po", "tendency_si", "tendency_sl"]
STRIKE_TENDENCY = ["strike_tendency_ch", "strike_tendency_cu", "strike_tendency_ff", "strike_tendency_po", "strike_tendency_si", "strike_tendency_sl"]
PITCHES_PERCENTAGES_DICT = {"CH": "percent_ch", "CU": "percent_cu", "EP": "percent_ep", "FC": "percent_fc", "FF": "percent_ff", "FO": "percent_fo", "FS": "percent_fs",
                       "KC": "percent_kc", "KN": "percent_kn", "PO": "percent_po", "SC": "percent_sc", "SI": "percent_si", "SL": "percent_sl"}
PITCHES_TENDENCY_DICT = {"CH": "tendency_ch", "CU": "tendency_cu", "FF": "tendency_ff", "PO": "tendency_po",
                         "SI": "tendency_si", "SL": "tendency_sl"}
STRIKE_TENDENCY_DICT = {"CH": "strike_tendency_ch", "CU": "strike_tendency_cu", "FF": "strike_tendency_ff",
                        "PO": "strike_tendency_po", "SI": "strike_tendency_si", "SL": "strike_tendency_sl"}
STRIKES_PERCENTAGES = ["percent_strikes_ch", "percent_strikes_cu", "percent_strikes_ep", "percent_strikes_fc", "percent_strikes_ff", "percent_strikes_fo", "percent_strikes_fs",
                       "percent_strikes_kc", "percent_strikes_kn", "percent_strikes_po", "percent_strikes_sc", "percent_strikes_si", "percent_strikes_sl"]
STRIKES_PERCENTAGES_DICT = {"CH": "percent_strikes_ch", "CU": "percent_strikes_cu", "EP": "percent_strikes_ep", "FC": "percent_strikes_fc", "FF": "percent_strikes_ff", "FO": "percent_strikes_fo", "FS": "percent_strikes_fs",
                       "KC": "percent_strikes_kc", "KN": "percent_strikes_kn", "PO": "percent_strikes_po", "SC": "percent_strikes_sc", "SI": "percent_strikes_si", "SL": "percent_strikes_sl"}
VERLANDER_ID = 434378
KERSHAW_ID = 477132
DEGROM_ID = 594798
BUMGARNER_ID = 518516
AVILA_ID = 488671

# def print_hi():
#     # Use a breakpoint in the code line below to debug your script.
#     data = statcast(start_dt='2017-06-24', end_dt='2018-10-29')
#     vals = data['inning_topbot']
#     types_of_pitches = set(vals)
#     # for i in range(len(data)):
#     #     row = data.iloc[i]
#     #     pitch_type = row['pitch_type']
#     #     types_of_pitches.add(pitch_type)
#     print(types_of_pitches)

TOP_BOT_LOWERCASE_MAP = {'top': 0, 'bot': 1}
MONTH_START = 5
MONTH_STOP = 7
DAY_START = 8
DAY_STOP = 10


def import_statcast_dataset():
    data = statcast(start_dt=START_DATE, end_dt=END_DATE)
    data.to_csv(path_or_buf='./statcast_dataset.csv', index=False)

def find_pitcher_catcher_combos():
    print("Reading data")
    # statcast_data = pd.read_csv(STATCAST_ORIGINAL_LOCATION_DATA)
    statcast_data = pd.read_pickle("./statcast_data.pickle")
    # with open('./statcast_data.pickle', 'wb') as handle:
    #     pickle.dump(statcast_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Bumgarner")
    num_of_pitches = len(statcast_data[(statcast_data['pitcher.1'] == BUMGARNER_ID)])
    print("Number of pitches,", num_of_pitches)
    pitcher_data = statcast_data.query("`pitcher.1` == @BUMGARNER_ID")
    print("Got pitcher data.")
    catchers = set(pitcher_data["fielder_2.1"].unique())
    assert(len(pitcher_data) == num_of_pitches)
    print("Got catchers data.")
    print("Format: catcher id, num of pitches with catcher")
    for catcher in catchers:
        print(catcher, ", ", len(pitcher_data[(pitcher_data['fielder_2.1'] == catcher)]))

def create_verlander_pickle():
    statcast_data = pd.read_pickle("./statcast_data.pickle")
    # with open('./statcast_data.pickle', 'wb') as handle:
    #     pickle.dump(statcast_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Verlander")
    num_of_pitches = len(statcast_data[(statcast_data['pitcher.1'] == VERLANDER_ID)])
    print("Number of pitches,", num_of_pitches)
    pitcher_data = statcast_data.query("`pitcher.1` == @VERLANDER_ID")
    assert (len(pitcher_data) == num_of_pitches)
    with open('./verlander_data.pickle', 'wb') as handle:
        pickle.dump(pitcher_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def clean_verlander_data():
   statcast_game_situation = pd.read_pickle("./verlander_data.pickle")
   print("Read pickle.")
   statcast_game_situation = statcast_game_situation.query("`fielder_2.1` == @AVILA_ID")
   print("Only Avila.")
   assert(len(statcast_game_situation) == 15155)

   # Convert game year from string to integer, and make the year 0-indexed starting at 2010.
   statcast_game_situation['game_year'] = statcast_game_situation['game_year'].astype("int")
   statcast_game_situation['game_year'] = statcast_game_situation['game_year'] - MIN_YEAR

   # Convert inning type from string to integer.
   statcast_game_situation['inning'] = statcast_game_situation['inning'].astype("int")

   # Convert balls from string to integer, and only keep rows where there's valid number of balls.
   statcast_game_situation['balls'] = statcast_game_situation['balls'].astype("int")
   statcast_game_situation = statcast_game_situation[statcast_game_situation['balls'] < 4]

   # Convert strikes from string to integer, and only keep rows where there's valid number of strikes.
   statcast_game_situation['strikes'] = statcast_game_situation['strikes'].astype("int")
   statcast_game_situation = statcast_game_situation[statcast_game_situation['strikes'] < 3]

   # Convert number of outs from string to integer.
   statcast_game_situation['outs_when_up'] = statcast_game_situation['outs_when_up'].astype("int")

   # Convert score of each team from string to integer, and store the difference.
   statcast_game_situation['bat_score'] = statcast_game_situation['bat_score'].astype("int")
   statcast_game_situation['fld_score'] = statcast_game_situation['fld_score'].astype("int")
   statcast_game_situation['score_diff'] = statcast_game_situation['fld_score'] - statcast_game_situation['bat_score']

   # Lowercase top and bottom of inning, then map top and bottom to 0 and 1.
   statcast_game_situation['inning_topbot'] = statcast_game_situation['inning_topbot'].str.lower()
   statcast_game_situation['inning_topbot'] = statcast_game_situation['inning_topbot'].map(TOP_BOT_LOWERCASE_MAP)

   # Store month and day from the game date
   statcast_game_situation['month'] = statcast_game_situation['game_date'].str.slice(MONTH_START, MONTH_STOP).astype(
       "int")
   statcast_game_situation['day'] = statcast_game_situation['game_date'].str.slice(DAY_START, DAY_STOP).astype(
       "int")

   # Convert FA to FF pitch type and CS to CU pitch type.
   statcast_game_situation = statcast_game_situation.replace('FA', 'FF')
   statcast_game_situation = statcast_game_situation.replace('CS', 'CU')
   statcast_game_situation = statcast_game_situation[statcast_game_situation['pitch_type'].isin(PITCHES)]

   # Set each base as a boolean where 0 if base is empty, and 1 if someone on base.
   statcast_game_situation['on_1b'] = (~statcast_game_situation['on_1b'].isnull()).astype("int")
   statcast_game_situation['on_2b'] = (~statcast_game_situation['on_2b'].isnull()).astype("int")
   statcast_game_situation['on_3b'] = (~statcast_game_situation['on_3b'].isnull()).astype("int")

   print("Making alignments")
   # One hot vector for out-fielding alignments.
   statcast_game_situation['of_std'] = (
           statcast_game_situation['of_fielding_alignment'].str.lower() == 'standard').astype("int")
   statcast_game_situation['of_strat'] = (
           statcast_game_situation['of_fielding_alignment'].str.lower() == 'strategic').astype("int")
   statcast_game_situation['of_extr'] = (
           statcast_game_situation['of_fielding_alignment'].str.lower() == 'extreme outfield shift').astype("int")
   statcast_game_situation['of_fourth'] = (
           statcast_game_situation['of_fielding_alignment'].str.lower() == '4th outfielder').astype("int")

   # One hot vector for in-fielding alignments.
   statcast_game_situation['if_std'] = (
           statcast_game_situation['if_fielding_alignment'].str.lower() == 'standard').astype("int")
   statcast_game_situation['if_strat'] = (
           statcast_game_situation['if_fielding_alignment'].str.lower() == 'strategic').astype("int")
   statcast_game_situation['if_shift'] = (
           statcast_game_situation['if_fielding_alignment'].str.lower() == 'infield shift').astype("int")
   print("Dropping columns")
   # Drop leftover columns
   statcast_game_situation = statcast_game_situation.drop(
       columns=['bat_score', 'fld_score', 'game_date', 'of_fielding_alignment', 'if_fielding_alignment'])

   # Organize the columns
   statcast_game_situation = statcast_game_situation.rename(columns={"game_year": "year", "outs_when_up": "outs"})

   print("Sorting.")
   statcast_game_situation = statcast_game_situation.sort_values(by = ['year', 'month', 'day', 'inning'], ascending=[True, True, True, True])

   print("Saving pickle.")
   with open('./verlander_avila_data.pickle', 'wb') as handle:
       pickle.dump(statcast_game_situation, handle, protocol=pickle.HIGHEST_PROTOCOL)
   print("Pickle saved.")


def clean_statcast_data():
    statcast_data = pd.read_csv(STATCAST_ORIGINAL_LOCATION_DATA)
    statcast_game_situation = statcast_data[
        ['pitch_type', 'game_year', 'game_date', 'inning', 'inning_topbot', 'balls', 'strikes', 'outs_when_up',
         'pitch_number', 'on_3b', 'on_2b', 'on_1b',
         'if_fielding_alignment', 'of_fielding_alignment', 'bat_score', 'fld_score']]

    # Convert game year from string to integer, and make the year 0-indexed starting at 2010.
    statcast_game_situation['game_year'] = statcast_game_situation['game_year'].astype("int")
    statcast_game_situation['game_year'] = statcast_game_situation['game_year'] - MIN_YEAR

    # Convert inning type from string to integer.
    statcast_game_situation['inning'] = statcast_game_situation['inning'].astype("int")

    # Convert balls from string to integer, and only keep rows where there's valid number of balls.
    statcast_game_situation['balls'] = statcast_game_situation['balls'].astype("int")
    statcast_game_situation = statcast_game_situation[statcast_game_situation['balls'] < 4]

    # Convert strikes from string to integer, and only keep rows where there's valid number of strikes.
    statcast_game_situation['strikes'] = statcast_game_situation['strikes'].astype("int")
    statcast_game_situation = statcast_game_situation[statcast_game_situation['strikes'] < 3]

    # Convert number of outs from string to integer.
    statcast_game_situation['outs_when_up'] = statcast_game_situation['outs_when_up'].astype("int")

    # Convert score of each team from string to integer, and store the difference.
    statcast_game_situation['bat_score'] = statcast_game_situation['bat_score'].astype("int")
    statcast_game_situation['fld_score'] = statcast_game_situation['fld_score'].astype("int")
    statcast_game_situation['score_diff'] = statcast_game_situation['fld_score'] - statcast_game_situation['bat_score']

    # Lowercase top and bottom of inning, then map top and bottom to 0 and 1.
    statcast_game_situation['inning_topbot'] = statcast_game_situation['inning_topbot'].str.lower()
    statcast_game_situation['inning_topbot'] = statcast_game_situation['inning_topbot'].map(TOP_BOT_LOWERCASE_MAP)

    # Store month from the game date
    statcast_game_situation['month'] = statcast_game_situation['game_date'].str.slice(MONTH_START, MONTH_STOP).astype(
        "int")

    # Convert FA to FF pitch type and CS to CU pitch type.
    statcast_game_situation = statcast_game_situation.replace('FA', 'FF')
    statcast_game_situation = statcast_game_situation.replace('CS', 'CU')
    statcast_game_situation = statcast_game_situation[statcast_game_situation['pitch_type'].isin(PITCHES)]

    # Set each base as a boolean where 0 if base is empty, and 1 if someone on base.
    statcast_game_situation['on_1b'] = (~statcast_game_situation['on_1b'].isnull()).astype("int")
    statcast_game_situation['on_2b'] = (~statcast_game_situation['on_2b'].isnull()).astype("int")
    statcast_game_situation['on_3b'] = (~statcast_game_situation['on_xf3b'].isnull()).astype("int")

    # One hot vector for out-fielding alignments.
    statcast_game_situation['of_std'] = (
            statcast_game_situation['of_fielding_alignment'].str.lower() == 'standard').astype("int")
    statcast_game_situation['of_strat'] = (
            statcast_game_situation['of_fielding_alignment'].str.lower() == 'strategic').astype("int")
    statcast_game_situation['of_extr'] = (
            statcast_game_situation['of_fielding_alignment'].str.lower() == 'extreme outfield shift').astype("int")
    statcast_game_situation['of_fourth'] = (
            statcast_game_situation['of_fielding_alignment'].str.lower() == '4th outfielder').astype("int")

    # One hot vector for in-fielding alignments.
    statcast_game_situation['if_std'] = (
            statcast_game_situation['if_fielding_alignment'].str.lower() == 'standard').astype("int")
    statcast_game_situation['if_strat'] = (
            statcast_game_situation['if_fielding_alignment'].str.lower() == 'strategic').astype("int")
    statcast_game_situation['if_shift'] = (
            statcast_game_situation['if_fielding_alignment'].str.lower() == 'infield shift').astype("int")

    # Drop leftover columns
    statcast_game_situation = statcast_game_situation.drop(
        columns=['bat_score', 'fld_score', 'game_date', 'of_fielding_alignment', 'if_fielding_alignment'])

    # Organize the columns
    statcast_game_situation = statcast_game_situation.rename(columns={"game_year": "year", "outs_when_up": "outs"})
    statcast_game_situation = statcast_game_situation[["pitch_type", "year", "month",
                                                       "score_diff", "inning", "inning_topbot",
                                                       "outs", "balls", "strikes", "pitch_number",
                                                       "on_1b", "on_2b", "on_3b",
                                                       "of_std", "of_strat", "of_extr", "of_fourth",
                                                       "if_std", "if_strat", "if_shift"]]

    print("Done cleaning")

    statcast_game_situation.to_csv("./cleaned_statcast_dataset.csv", index=False)

    print("Done saving cleaned data")


def prep_train_test_data():
    cleaned_data = pd.read_csv("./cleaned_statcast_dataset.csv")

    y = cleaned_data["pitch_type"].to_numpy()
    X = cleaned_data.loc[:, cleaned_data.columns != 'pitch_type'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    pd.DataFrame(X_train).to_csv("./baseline_X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("./baseline_X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("./baseline_y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("./baseline_y_test.csv", index=False)


def train_neural_network(hidden_layer_sizes=(250, 300, 300, 250), early_stopping=True, max_iter=40, n_iter_no_change=3):
    x_train = pd.read_csv("./baseline_X_train.csv").to_numpy()
    x_train = x_train[:100000]
    x_test = pd.read_csv("./baseline_X_test.csv").to_numpy()
    x_test = x_test[:100000]
    y_train = pd.read_csv("./baseline_y_train.csv").to_numpy().reshape(-1)
    y_train = y_train[:100000]
    y_test = pd.read_csv("./baseline_y_test.csv").to_numpy().reshape(-1)
    y_test = y_test[:100000]
    print("Starting classifier")
    baseline_nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, early_stopping=early_stopping,
                                n_iter_no_change=n_iter_no_change, max_iter=max_iter, verbose=True)
    print("Begin fitting")
    baseline_nn.fit(x_train, y_train)
    print("Done fitting")
    with open('./baseline_nn.pickle', 'wb') as handle:
        pickle.dump(baseline_nn, handle, protocol=pickle.HIGHEST_PROTOCOL)
    accuracy = baseline_nn.score(x_test, y_test)
    print(accuracy)
    # accuracy = accuracy_score(y_test, preds)
    # balanced_accuracy = balanced_accuracy_score(y_test, preds)
    # top_k_accuracy = top_k_accuracy_score(y_test, preds, k=2)
    # avg_precision = average_precision_score(y_test, preds)
    # roc_auc = roc_auc_score(y_test, preds)
    # print("accuracy: ", accuracy)
    # print("balanced accuracy: ", balanced_accuracy)
    # print("top 2 accuracy: ", top_k_accuracy)
    # print("avg precision: ", avg_precision)
    # print("roc auc: ", roc_auc)


# def get_sizes():
    # cleaned_data = pd.read_csv("./cleaned_statcast_dataset.csv")
    # print(len(cleaned_data[(cleaned_data['pitcher'] == 425794) and (cleaned_data['fielder_2'] == '426877')]))
    # x_train = pd.read_csv("./baseline_X_train.csv").to_numpy()
    # x_test = pd.read_csv("./baseline_X_test.csv").to_numpy()
    # y_train = pd.read_csv("./baseline_y_train.csv").to_numpy().reshape(-1)
    # y_test = pd.read_csv("./baseline_y_test.csv").to_numpy().reshape(-1)
    # print(cleaned_data.shape, x_train.shape, y_test.shape, y_train.shape, y_test.shape)

def analyze_verlander_avila_pickle():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    pitches = pd.read_pickle("./verlander_avila_data.pickle")
    # events = pitches['events'].unique()
    # pitches = pitches[["pitch_type", "year", "month","day",
    #                                                    "score_diff", "inning", "inning_topbot",
    #                                                    "outs", "balls", "strikes", "pitch_number",
    #                                                    "on_1b", "on_2b", "on_3b",
    #                                                    "of_std", "of_strat", "of_extr", "of_fourth",
    #                                                    "if_std", "if_strat", "if_shift", "at_bat_number", "type", "events"]]
    #
    #
    # pitches = pitches.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
    #                                                               ascending=[True, True, True, True, True, True])
    # n_pitches = pitches.head(20)
    rslt_df = (pitches[pitches['events'] == 'other_out']).head(1)
    print(rslt_df['description'])
    # grouped_df = n_pitches.groupby(by=['year', 'month', 'day'])
    # sorted = grouped_df.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
    #                                                               ascending=[True, True, True, True, True, True])
    # l_grouped = list(grouped_df)
    # print(l_grouped[0][1])
    # print(n_pitches)

def get_verlander_and_avila_data():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    pitches = pd.read_pickle("./verlander_avila_data.pickle")

    pitches = pitches[["pitch_type", "year", "month","day",
                                                       "score_diff", "inning", "inning_topbot",
                                                       "outs", "balls", "strikes", "pitch_number",
                                                       "on_1b", "on_2b", "on_3b",
                                                       "of_std", "of_strat", "of_extr", "of_fourth",
                                                       "if_std", "if_strat", "if_shift", "at_bat_number", "type"]]
    pitches = pitches.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
                                                                  ascending=[True, True, True, True, True, True])
    # n_pitches = pitches.head(21)
    # second_game = pitches.iloc[200:221, :]
    # n_pitches = pd.concat([n_pitches, second_game], axis=0)
    # n_pitches = n_pitches.reset_index(drop=True)
    return pitches


def get_type_percentages(pitches):
    # pitches = pd.read_pickle("./verlander_avila_data.pickle")
    # pitches = pitches[["pitch_type", "year", "month", "day",
    #                    "score_diff", "inning", "inning_topbot",
    #                    "outs", "balls", "strikes", "pitch_number",
    #                    "on_1b", "on_2b", "on_3b",
    #                    "of_std", "of_strat", "of_extr", "of_fourth",
    #                    "if_std", "if_strat", "if_shift", "at_bat_number"]]
    # pitches = pitches.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
    #                               ascending=[True, True, True, True, True, True])
    # pitches = pitches.head(20)
    prev_result_list = ['prev_pitch_strike', 'prev_pitch_ball', 'prev_pitch_in_play']
    zeros = np.zeros(shape=(len(pitches), len(PITCHES_LIST) + len(prev_result_list)))
    column_names = PITCHES_PERCENTAGES + prev_result_list
    pitch_percentages = pd.DataFrame(zeros, columns=column_names)
    pitch_type_counts = {}
    for pitch in PITCHES_LIST:
        pitch_type_counts[pitch] = 0
    num_pitches = 0
    for i in range(1, len(pitches)):
        num_pitches += 1
        previous_pitch = pitches.iloc[i-1]['pitch_type']
        previous_result = pitches.iloc[i-1]['type']
        pitch_type_counts[previous_pitch] += 1
        prev_results_dict = {}
        assert(previous_result in {'S', 'B', 'X'})
        prev_results_dict['prev_pitch_strike'] = 1 if previous_result == 'S' else 0
        prev_results_dict['prev_pitch_ball'] = 1 if previous_result == 'B' else 0
        prev_results_dict['prev_pitch_in_play'] = 1 if previous_result == 'X' else 0
        for pitch in PITCHES_LIST:
            percentage = pitch_type_counts[pitch] / num_pitches
            col_to_update = PITCHES_PERCENTAGES_DICT[pitch]
            pitch_percentages.iloc[i][col_to_update] = percentage
        for key in prev_results_dict:
            pitch_percentages.iloc[i][key] = prev_results_dict[key]
    return pitch_percentages


def get_strike_percentages(pitches):
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 150)
    # pitches = pd.read_pickle("./verlander_avila_data.pickle")
    # pitches = pitches[["pitch_type", "year", "month", "day",
    #                    "score_diff", "inning", "inning_topbot",
    #                    "outs", "balls", "strikes", "pitch_number",
    #                    "on_1b", "on_2b", "on_3b",
    #                    "of_std", "of_strat", "of_extr", "of_fourth",
    #                    "if_std", "if_strat", "if_shift", "at_bat_number", "type"]]
    # pitches = pitches.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
    #                               ascending=[True, True, True, True, True, True])
    # pitches = pitches.head(20)
    # print(pitches)
    zeros = np.zeros(shape=(len(pitches), len(PITCHES_LIST)))
    strike_percentages = pd.DataFrame(zeros, columns=STRIKES_PERCENTAGES)
    pitch_type_counts = {}
    strike_counts = {}
    for pitch in PITCHES_LIST:
        pitch_type_counts[pitch] = 0
        strike_counts[pitch] = 0
    for i in range(1, len(pitches)):
        previous_pitch = pitches.iloc[i-1]['pitch_type']
        previous_result = pitches.iloc[i-1]['type']
        pitch_type_counts[previous_pitch] += 1
        if previous_result == 'S':
            strike_counts[previous_pitch] += 1
        for pitch in PITCHES_LIST:
            percentage = 0
            if pitch_type_counts[pitch] > 0:
                percentage = strike_counts[pitch] / pitch_type_counts[pitch]
            col_to_update = STRIKES_PERCENTAGES_DICT[pitch]
            strike_percentages.iloc[i][col_to_update] = percentage
    return strike_percentages


def get_previous_strike_tendencies(pitches):
    # pitches = pd.read_pickle("./verlander_avila_data.pickle")
    # pitches = pitches[["pitch_type", "year", "month", "day",
    #                    "score_diff", "inning", "inning_topbot",
    #                    "outs", "balls", "strikes", "pitch_number",
    #                    "on_1b", "on_2b", "on_3b",
    #                    "of_std", "of_strat", "of_extr", "of_fourth",
    #                    "if_std", "if_strat", "if_shift", "at_bat_number"]]
    # pitches = pitches.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
    #                               ascending=[True, True, True, True, True, True])
    # pitches = pitches.head(20)
    grouped_df = pitches.groupby(by=['year', 'month', 'day'])
    l_grouped = list(grouped_df)
    last_n = [5, 10, 20]
    games = []
    for group in l_grouped:
        data = group[1]
        game = []
        for n in last_n:
            zeros = np.zeros(shape=(len(data), len(STRIKE_TENDENCY)))
            column_name = []
            for pitch in STRIKE_TENDENCY:
                column_name.append(pitch + '_' + str(n))
            strike_tendencies = pd.DataFrame(zeros, columns=column_name)
            strike_type_counts = {}
            for pitch in VERLANDER_PITCHES_LIST:
                strike_type_counts[pitch] = [[], 0]
            for i in range(1, len(data)):
                previous_pitch = data.iloc[i-1]['pitch_type']
                previous_result = data.iloc[i - 1]['type']
                strike_type_counts[previous_pitch][0].append(previous_result)
                if previous_result == 'S':
                    strike_type_counts[previous_pitch][1] += 1
                if len(strike_type_counts[previous_pitch][0]) > n:
                    first_pitch = strike_type_counts[previous_pitch][0][0]
                    if first_pitch == 'S':
                        strike_type_counts[previous_pitch][1] -= 1
                    updated_pitches = strike_type_counts[previous_pitch][0][1:]
                    strike_type_counts[previous_pitch][0] = updated_pitches
                for pitch in VERLANDER_PITCHES_LIST:
                    num_throws = len(strike_type_counts[pitch][0])
                    assert(num_throws <= n)
                    num_strikes = strike_type_counts[pitch][1]
                    assert (num_strikes <= num_throws)
                    percentage = 0
                    if num_throws > 0:
                        percentage = num_strikes / num_throws
                    col_to_update = STRIKE_TENDENCY_DICT[pitch]
                    col_to_update = col_to_update + '_' + str(n)
                    strike_tendencies.iloc[i][col_to_update] = percentage
            game.append(strike_tendencies)
        game_total = pd.concat(game, axis=1)
        games.append(game_total)
    games_total = pd.concat(games, axis=0)
    return games_total

def get_previous_pitch_tendencies(pitches):
    # pitches = pd.read_pickle("./verlander_avila_data.pickle")
    # pitches = pitches[["pitch_type", "year", "month", "day",
    #                    "score_diff", "inning", "inning_topbot",
    #                    "outs", "balls", "strikes", "pitch_number",
    #                    "on_1b", "on_2b", "on_3b",
    #                    "of_std", "of_strat", "of_extr", "of_fourth",
    #                    "if_std", "if_strat", "if_shift", "at_bat_number"]]
    # pitches = pitches.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
    #                               ascending=[True, True, True, True, True, True])
    # pitches = pitches.head(20)
    grouped_df = pitches.groupby(by=['year', 'month', 'day'])
    l_grouped = list(grouped_df)
    last_n = [1, 5, 10, 20]
    games = []
    for group in l_grouped:
        data = group[1]
        game = []
        for n in last_n:
            zeros = np.zeros(shape=(len(data), len(PITCHES_TENDENCY)))
            column_name = []
            for pitch in PITCHES_TENDENCY:
                column_name.append(pitch + '_' + str(n))
            pitch_tendencies = pd.DataFrame(zeros, columns=column_name)
            pitch_type_counts = {}
            for pitch in VERLANDER_PITCHES_LIST:
                pitch_type_counts[pitch] = 0
            last_n_pitches = []
            for i in range(1, len(data)):
                previous_pitch = data.iloc[i-1]['pitch_type']
                last_n_pitches.append(previous_pitch)
                pitch_type_counts[previous_pitch] += 1
                if len(last_n_pitches) > n:
                    first_pitch = last_n_pitches[0]
                    last_n_pitches = last_n_pitches[1:]
                    pitch_type_counts[first_pitch] -= 1
                num_pitches = len(last_n_pitches)
                percentage_sum = 0
                for pitch in VERLANDER_PITCHES_LIST:
                    percentage = pitch_type_counts[pitch] / num_pitches
                    assert(percentage <= 1)
                    percentage_sum += percentage
                    col_to_update = PITCHES_TENDENCY_DICT[pitch]
                    col_to_update = col_to_update + '_' + str(n)
                    pitch_tendencies.iloc[i][col_to_update] = percentage
            game.append(pitch_tendencies)
        game_total = pd.concat(game, axis=1)
        games.append(game_total)
    games_total = pd.concat(games, axis=0)
    return games_total

def merge_pitching_data(pitches, pitch_percentages, strike_percentages):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    # pitches = pd.read_pickle("./verlander_avila_data.pickle")
    # pitches = pitches[["pitch_type", "year", "month", 'day',
    #                                                    "score_diff", "inning", "inning_topbot",
    #                                                    "outs", "balls", "strikes", "pitch_number",
    #                                                    "on_1b", "on_2b", "on_3b",
    #                                                    "of_std", "of_strat", "of_extr", "of_fourth",
    #                                                    "if_std", "if_strat", "if_shift", "at_bat_number"]]
    # pitches = pitches.sort_values(by=['year', 'month', 'day', 'inning', 'at_bat_number', 'pitch_number'],
    #                               ascending=[True, True, True, True, True, True])
    # pitches = pitches.head(20)
    # pitches = pitches.reset_index(drop=True)
    frames = [pitches, pitch_percentages, strike_percentages]
    result = pd.concat(frames, axis=1)
    return result



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cache.enable()
    # import_statcast_dataset()
    # clean_statcast_data()
    # create_verlander_pickle()
    # prep_train_test_data()
    # train_neural_network()
    # get_sizes()
    # clean_verlander_data()
    # analyze_verlander_avila_pickle()
    pitch_data = get_verlander_and_avila_data()
    # print(pitch_data[['pitch_type', 'type']])
    # print(pitch_data)
    result = get_previous_pitch_tendencies(pitch_data)
    print(len(result))
    # result = get_previous_strike_tendencies(pitch_data)
    # print(len(result))
    # print(result.iloc[len(result) - 1])
    type_percentages = get_type_percentages(pitch_data)
    print(len(type_percentages))
    # strike_percentages = get_strike_percentages(pitch_data)
    # print(len(type_percentages), len(strike_percentages))
    # data = merge_pitching_data(pitch_data, type_percentages, strike_percentages)
    # print(data.iloc[len(data) - 1])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
