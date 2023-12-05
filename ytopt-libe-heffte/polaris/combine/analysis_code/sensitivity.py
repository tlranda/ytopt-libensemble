import numpy as np
import pandas as pd
import shap
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--csv', nargs='+', default=None, help="Files to concatenate and do sensitivity analysis on")
    prs.add_argument('--summarize', action='store_true', help="Combine all files (default: treat separately)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    if type(args.csv) is str:
        args.csv = [args.csv]
    return args

def convert_frame(frame):
    param_names = np.asarray([f'p{_}' for _ in range(10)] + ['c0', 'mpi_ranks'])
    y_name = np.asarray(["FLOPS"])
    frame_X, frame_Y = frame[param_names], frame[y_name]
    # Scikit-Learn will only operate on pure float data
    index = frame_X.index
    frame_X.loc[index,'p0'] = [0. if _ == 'double' else 1. for _ in frame_X['p0']]
    p2_list = ['-no-reorder', '-reorder', ' ']
    p2_len = len(p2_list)
    frame_X.loc[index,'p2'] = [p2_list.index(_)/p2_len for _ in frame_X['p2']]
    p3_list = ['-a2a', '-a2av', ' ']
    p3_len = len(p3_list)
    frame_X.loc[index,'p3'] = [p3_list.index(_)/p3_len for _ in frame_X['p3']]
    p4_list = ['-p2p', '-p2p_pl', ' ']
    p4_len = len(p4_list)
    frame_X.loc[index,'p4'] = [p4_list.index(_)/p4_len for _ in frame_X['p4']]
    p5_list = ['-pencils', '-slabs', ' ']
    p5_len = len(p5_list)
    frame_X.loc[index,'p5'] = [p5_list.index(_)/p5_len for _ in frame_X['p5']]
    p6_list = ['-r2c_dir 0', '-r2c_dir 1', '-r2c_dir 2', ' ']
    p6_len = len(p6_list)
    frame_X.loc[index,'p6'] = [p6_list.index(_)/p6_len for _ in frame_X['p6']]
    c0_list = ['cufft', 'fftw']
    c0_len = len(c0_list)
    frame_X.loc[index,'c0'] = [c0_list.index(_)/c0_len for _ in frame_X['c0']]
    return param_names, frame_X, frame_Y

def main(args=None):
    args = parse(args)
    # Loading
    csvs = []
    for csv in args.csv:
        try:
            csvs.append(pd.read_csv(csv))
        except FileNotFoundError:
            print(f"Could not open file {csv}")
    if args.summarize:
        data = [pd.concat(csvs).reset_index(drop=True)]
        names = [f'concatenated: {args.csv}']
    else:
        data = csvs
        names = args.csv
    for loaded, name in zip(data, names):
        print(name)
        param_names, load_X, load_Y = convert_frame(loaded)
        # Scikit-Learn R2 score / regressor for importance
        regressor = RandomForestRegressor()
        X_train, X_test, Y_train, Y_test = train_test_split(load_X, load_Y.to_numpy().ravel(),
                                                            test_size=0.5,
                                                            random_state=42)
        regressor.fit(X_train, Y_train)
        predictions = regressor.predict(load_X)
        r2 = r2_score(load_Y, predictions)
        print(f"R2 score of RandomForest: {r2}")

        # Scikit-Learn importance
        regression_feature_importance = regressor.feature_importances_
        sklearn_importance = regression_feature_importance.argsort()
        print("\nRandomForest feature priority:\n",param_names[sklearn_importance])
        print("Importances:\n",regression_feature_importance[sklearn_importance])
        print("Ranking:\n",sklearn_importance)

        # SHAP importance
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(load_X)
        shap_accumulate = shap_values.sum(axis=0)
        # Shap importance is not monotonic -- negative values just mean different classification
        shap_importance = np.abs(shap_accumulate).argsort()
        print("\nShap feature priority:\n",param_names[shap_importance])
        print("Importances:\n",shap_accumulate[shap_importance])
        print("Ranking:\n",shap_importance)

        # Combined importance is misleading and hard to do correctly
        #combined = sklearn_importance + shap_importance
        #combined_importance = combined.argsort()
        #print("Combined feature priority:\n",param_names[combined_importance])
        #print("Ranking based on:\n",combined)
        #print("Ranking:\n",combined_importance)

        print()

if __name__ == '__main__':
    main()

