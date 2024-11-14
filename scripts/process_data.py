import sys
import os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))

import argparse

import numpy as np
import pandas as pd
import torch
import sklearn.cluster

from src.data.processing import *

def process_data(windowsize, upperRUL):
    # TODO: only preprocess selected datasets
    # TODO: padding when windowsize < length of test segment

    if isinstance(windowsize, int):
        windowsize = [windowsize]*4
    else:
        assert len(windowsize) == 4

    header = ["unit", "cycles", "operational setting 1", "operational setting 2", "operational setting 3"] + [f"sensor {i}" for i in range(1,21+1)]
    drop_cols = [f'sensor {i}' for i in [1,5,6,10,16,18,19]]
    ids = [f'sensor {i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
    op_settings = [f'operational setting {i}' for i in range(1,4)]

    dataFD001_pd = pd.read_csv(data_dir+"/raw/train_FD001.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)
    dataFD002_pd = pd.read_csv(data_dir+"/raw/train_FD002.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)
    dataFD003_pd = pd.read_csv(data_dir+"/raw/train_FD003.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)
    dataFD004_pd = pd.read_csv(data_dir+"/raw/train_FD004.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)

    testdataFD001_pd = pd.read_csv(data_dir+"/raw/test_FD001.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)
    testdataFD002_pd = pd.read_csv(data_dir+"/raw/test_FD002.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)
    testdataFD003_pd = pd.read_csv(data_dir+"/raw/test_FD003.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)
    testdataFD004_pd = pd.read_csv(data_dir+"/raw/test_FD004.txt", sep=" ", header=None, usecols=[i for i in range(26)], names=header).drop(columns=drop_cols)

    testrulFD001_pd = pd.read_csv(data_dir+"/raw/RUL_FD001.txt", header=None, names=['RUL'])
    testrulFD002_pd = pd.read_csv(data_dir+"/raw/RUL_FD002.txt", header=None, names=['RUL'])
    testrulFD003_pd = pd.read_csv(data_dir+"/raw/RUL_FD003.txt", header=None, names=['RUL'])
    testrulFD004_pd = pd.read_csv(data_dir+"/raw/RUL_FD004.txt", header=None, names=['RUL'])

    dataFD001_normed = dataFD001_pd.copy()
    dataFD002_normed = dataFD002_pd.copy()
    dataFD003_normed = dataFD003_pd.copy()
    dataFD004_normed = dataFD004_pd.copy()

    testdataFD001_normed = testdataFD001_pd.copy()
    testdataFD002_normed = testdataFD002_pd.copy()
    testdataFD003_normed = testdataFD003_pd.copy()
    testdataFD004_normed = testdataFD004_pd.copy()

    # cluster operational settings
    opdata = np.concatenate([dataFD001_normed[op_settings],
                             dataFD002_normed[op_settings],
                             dataFD003_normed[op_settings],
                             dataFD004_normed[op_settings]])
    kmeans = sklearn.cluster.KMeans(n_clusters=6, random_state=0, n_init='auto').fit(opdata)
    dataFD001_normed['operating condition'] = kmeans.predict(dataFD001_normed[op_settings])+1
    dataFD002_normed['operating condition'] = kmeans.predict(dataFD002_normed[op_settings])+1
    dataFD003_normed['operating condition'] = kmeans.predict(dataFD003_normed[op_settings])+1
    dataFD004_normed['operating condition'] = kmeans.predict(dataFD004_normed[op_settings])+1
    testdataFD001_normed['operating condition'] = kmeans.predict(testdataFD001_normed[op_settings])+1
    testdataFD002_normed['operating condition'] = kmeans.predict(testdataFD002_normed[op_settings])+1
    testdataFD003_normed['operating condition'] = kmeans.predict(testdataFD003_normed[op_settings])+1
    testdataFD004_normed['operating condition'] = kmeans.predict(testdataFD004_normed[op_settings])+1
    

    for i in range(1,7):
        dataFD001_normed[f'oc{i} count'] = np.cumsum(dataFD001_normed['operating condition']==i)
        dataFD002_normed[f'oc{i} count'] = np.cumsum(dataFD002_normed['operating condition']==i)
        dataFD003_normed[f'oc{i} count'] = np.cumsum(dataFD003_normed['operating condition']==i)
        dataFD004_normed[f'oc{i} count'] = np.cumsum(dataFD004_normed['operating condition']==i)
        testdataFD001_normed[f'oc{i} count'] = np.cumsum(testdataFD001_normed['operating condition']==i)
        testdataFD002_normed[f'oc{i} count'] = np.cumsum(testdataFD002_normed['operating condition']==i)
        testdataFD003_normed[f'oc{i} count'] = np.cumsum(testdataFD003_normed['operating condition']==i)
        testdataFD004_normed[f'oc{i} count'] = np.cumsum(testdataFD004_normed['operating condition']==i)


    # normalise data sets per operating condition
    for i in range(1,7): # loop over OCs
        testmask = testdataFD001_normed['operating condition'] == i
        mask = dataFD001_normed['operating condition'] == i
        testdataFD001_normed.loc[np.where(testmask)[0],ids] = (testdataFD001_normed[ids][testmask] - dataFD001_normed[ids][mask].min()) / (dataFD001_normed[ids][mask].max() - dataFD001_normed[ids][mask].min())
        dataFD001_normed.loc[np.where(mask)[0],ids] = (dataFD001_normed[ids][mask] - dataFD001_normed[ids][mask].min()) / (dataFD001_normed[ids][mask].max() - dataFD001_normed[ids][mask].min())
        testmask = testdataFD002_normed['operating condition'] == i
        mask = dataFD002_normed['operating condition'] == i
        testdataFD002_normed.loc[np.where(testmask)[0],ids] = (testdataFD002_normed[ids][testmask] - dataFD002_normed[ids][mask].min()) / (dataFD002_normed[ids][mask].max() - dataFD002_normed[ids][mask].min())
        dataFD002_normed.loc[np.where(mask)[0],ids] = (dataFD002_normed[ids][mask] - dataFD002_normed[ids][mask].min()) / (dataFD002_normed[ids][mask].max() - dataFD002_normed[ids][mask].min())
        testmask = testdataFD003_normed['operating condition'] == i
        mask = dataFD003_normed['operating condition'] == i
        testdataFD003_normed.loc[np.where(testmask)[0],ids] = (testdataFD003_normed[ids][testmask] - dataFD003_normed[ids][mask].min()) / (dataFD003_normed[ids][mask].max() - dataFD003_normed[ids][mask].min())
        dataFD003_normed.loc[np.where(mask)[0],ids] = (dataFD003_normed[ids][mask] - dataFD003_normed[ids][mask].min()) / (dataFD003_normed[ids][mask].max() - dataFD003_normed[ids][mask].min())
        testmask = testdataFD004_normed['operating condition'] == i
        mask = dataFD004_normed['operating condition'] == i
        testdataFD004_normed.loc[np.where(testmask)[0],ids] = (testdataFD004_normed[ids][testmask] - dataFD004_normed[ids][mask].min()) / (dataFD004_normed[ids][mask].max() - dataFD004_normed[ids][mask].min())
        dataFD004_normed.loc[np.where(mask)[0],ids] = (dataFD004_normed[ids][mask] - dataFD004_normed[ids][mask].min()) / (dataFD004_normed[ids][mask].max() - dataFD004_normed[ids][mask].min())


    # add RUL columns to train dataset 
    dataFD001_normed.insert(2, "RUL", get_ruls(dataFD001_normed))
    dataFD002_normed.insert(2, "RUL", get_ruls(dataFD002_normed))
    dataFD003_normed.insert(2, "RUL", get_ruls(dataFD003_normed))
    dataFD004_normed.insert(2, "RUL", get_ruls(dataFD004_normed))

    dataFD001_normed.insert(2, "RUL_pw", np.minimum(dataFD001_normed['RUL'].values, upperRUL))
    dataFD002_normed.insert(2, "RUL_pw", np.minimum(dataFD002_normed['RUL'].values, upperRUL))
    dataFD003_normed.insert(2, "RUL_pw", np.minimum(dataFD003_normed['RUL'].values, upperRUL))
    dataFD004_normed.insert(2, "RUL_pw", np.minimum(dataFD004_normed['RUL'].values, upperRUL))

    testdataFD001_normed.insert(2, "RUL", 
                                np.repeat(testrulFD001_pd['RUL'].values, testdataFD001_normed['unit'].value_counts(sort=False))+np.concatenate([np.arange(0,v)[::-1] for v in testdataFD001_normed['unit'].value_counts(sort=False)]))
    testdataFD002_normed.insert(2, "RUL", 
                                np.repeat(testrulFD002_pd['RUL'].values, testdataFD002_normed['unit'].value_counts(sort=False))+np.concatenate([np.arange(0,v)[::-1] for v in testdataFD002_normed['unit'].value_counts(sort=False)]))
    testdataFD003_normed.insert(2, "RUL", 
                                np.repeat(testrulFD003_pd['RUL'].values, testdataFD003_normed['unit'].value_counts(sort=False))+np.concatenate([np.arange(0,v)[::-1] for v in testdataFD003_normed['unit'].value_counts(sort=False)]))
    testdataFD004_normed.insert(2, "RUL", 
                                np.repeat(testrulFD004_pd['RUL'].values, testdataFD004_normed['unit'].value_counts(sort=False))+np.concatenate([np.arange(0,v)[::-1] for v in testdataFD004_normed['unit'].value_counts(sort=False)]))

    testdataFD001_normed.insert(2, "RUL_pw", np.minimum(testdataFD001_normed['RUL'].values, upperRUL))
    testdataFD002_normed.insert(2, "RUL_pw", np.minimum(testdataFD002_normed['RUL'].values, upperRUL))
    testdataFD003_normed.insert(2, "RUL_pw", np.minimum(testdataFD003_normed['RUL'].values, upperRUL))
    testdataFD004_normed.insert(2, "RUL_pw", np.minimum(testdataFD004_normed['RUL'].values, upperRUL))

    # convert to torch tensors
    engineid_FD001, RULs_FD001, data_FD001, RULspw_FD001 = create_torch_tensors(dataFD001_normed, windowsize[0], ['operating condition']+ids)
    engineid_FD002, RULs_FD002, data_FD002, RULspw_FD002 = create_torch_tensors(dataFD002_normed, windowsize[1], ['operating condition']+ids)
    engineid_FD003, RULs_FD003, data_FD003, RULspw_FD003 = create_torch_tensors(dataFD003_normed, windowsize[2], ['operating condition']+ids)
    engineid_FD004, RULs_FD004, data_FD004, RULspw_FD004 = create_torch_tensors(dataFD004_normed, windowsize[3], ['operating condition']+ids)

    testengineid_FD001, testRULs_FD001, testdata_FD001, testRULspw_FD001 = create_torch_tensors(testdataFD001_normed, windowsize[0], ['operating condition']+ids)
    testengineid_FD002, testRULs_FD002, testdata_FD002, testRULspw_FD002 = create_torch_tensors(testdataFD002_normed, windowsize[1], ['operating condition']+ids)
    testengineid_FD003, testRULs_FD003, testdata_FD003, testRULspw_FD003 = create_torch_tensors(testdataFD003_normed, windowsize[2], ['operating condition']+ids)
    testengineid_FD004, testRULs_FD004, testdata_FD004, testRULspw_FD004 = create_torch_tensors(testdataFD004_normed, windowsize[3], ['operating condition']+ids)

    # custom datasets
    FD001_dataset = CMAPSS_dataset(engineid_FD001, RULs_FD001, RULspw_FD001, data_FD001, 'FD001')
    FD002_dataset = CMAPSS_dataset(engineid_FD002, RULs_FD002, RULspw_FD002, data_FD002, 'FD002')
    FD003_dataset = CMAPSS_dataset(engineid_FD003, RULs_FD003, RULspw_FD003, data_FD003, 'FD003')
    FD004_dataset = CMAPSS_dataset(engineid_FD004, RULs_FD004, RULspw_FD004, data_FD004, 'FD004')

    FD001_dataset_test = CMAPSS_dataset(testengineid_FD001, testRULs_FD001, testRULspw_FD001, testdata_FD001, 'FD001-test')
    FD002_dataset_test = CMAPSS_dataset(testengineid_FD002, testRULs_FD002, testRULspw_FD002, testdata_FD002, 'FD002-test')
    FD003_dataset_test = CMAPSS_dataset(testengineid_FD003, testRULs_FD003, testRULspw_FD003, testdata_FD003, 'FD003-test')
    FD004_dataset_test = CMAPSS_dataset(testengineid_FD004, testRULs_FD004, testRULspw_FD004, testdata_FD004, 'FD004-test')

    # save torch datasets
    torch.save(FD001_dataset, data_dir+f'/processed/trainsetFD001_w{windowsize[0]}_M{upperRUL}.pt')
    torch.save(FD002_dataset, data_dir+f'/processed/trainsetFD002_w{windowsize[1]}_M{upperRUL}.pt')
    torch.save(FD003_dataset, data_dir+f'/processed/trainsetFD003_w{windowsize[2]}_M{upperRUL}.pt')
    torch.save(FD004_dataset, data_dir+f'/processed/trainsetFD004_w{windowsize[3]}_M{upperRUL}.pt')

    torch.save(FD001_dataset_test, data_dir+f'/processed/testsetFD001_w{windowsize[0]}_M{upperRUL}.pt')
    torch.save(FD002_dataset_test, data_dir+f'/processed/testsetFD002_w{windowsize[1]}_M{upperRUL}.pt')
    torch.save(FD003_dataset_test, data_dir+f'/processed/testsetFD003_w{windowsize[2]}_M{upperRUL}.pt')
    torch.save(FD004_dataset_test, data_dir+f'/processed/testsetFD004_w{windowsize[3]}_M{upperRUL}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--windowsize', type=int, default=19, required=False,
                        help='Size of the sliding window in cycles (default: %(default)s)')
    parser.add_argument('-u', '--upperRUL', type=int, default=128, required=False,
                        help='Upper value for the piecewise RUL (default: %(default)s)')
    
    args = parser.parse_args()
    
    process_data(args.windowsize, args.upperRUL)
    