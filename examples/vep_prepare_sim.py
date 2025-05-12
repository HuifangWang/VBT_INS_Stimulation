#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:49:44 2019

@author: Huifang Wang  adapted from retrospective patients to trial patients
"""
import mne
import sys
import scipy.signal as signal
import numpy as np
import zipfile
import os
import glob
import multiprocessing as mp
import scipy.signal
from scipy.ndimage import binary_erosion
import os.path as op
import matplotlib.pyplot as plt

def read_vep_mrtrix_lut():
    roi_names = []
#     with open('../util/data/VepMrtrixLut.txt', 'r') as fd:
    with open('/root/capsule/code/util/data/VepMrtrixLut.txt', 'r') as fd:
        for line in fd.readlines():
            i, roi_name, *_ = line.strip().split()
            roi_names.append(roi_name)
            #roi_name_to_index[roi_name.lower()] = int(i) - 1
    roi=roi_names[1:]
    return roi

def bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names, is_minus=True):
    from icdc import seeg_ch_name_split
    split_names = [seeg_ch_name_split(_) for _ in seeg_xyz_names]
    bip_gain_rows = []
    bip_xyz = []
    bip_names = []
    for i in range(len(split_names) - 1):
        try:
            name, idx = split_names[i]
            next_name, next_idx = split_names[i + 1]
            if name == next_name:
                if is_minus:
                    bip_gain_rows.append(gain[i + 1] - gain[i])
                else:
                    bip_gain_rows.append((gain[i + 1] + gain[i]) / 2.0)
                bip_xyz.append(
                    [(p + q) / 2.0 for p, q in zip(seeg_xyz[i][1], seeg_xyz[i + 1][1])]
                )
                bip_names.append("%s%d-%d" % (name, idx, next_idx))
        except Exception as exc:
            print(exc)
    # abs val, envelope/power always postive
    bip_gain = np.abs(np.array(bip_gain_rows))
    bip_xyz = np.array(bip_xyz)
    return bip_gain, bip_xyz, bip_names

def _bipify_raw(raw):
    from icdc import seeg_ch_name_split
    split_names = [seeg_ch_name_split(_) for _ in raw.ch_names]
    bip_ch_names = []
    bip_ch_data = []
    for i in range(len(split_names) - 1):
        try:
            name, idx = split_names[i]
            next_name, next_idx = split_names[i + 1]
            if name == next_name:
                bip_ch_names.append('%s%d-%d' % (name, idx, next_idx))
                data, _ = raw[[i, i + 1]]
                bip_ch_data.append(data[1] - data[0])
        except:
            pass
    info = mne.create_info(
        ch_names=bip_ch_names,
        sfreq=raw.info['sfreq'],
        ch_types=['eeg' for _ in bip_ch_names])
    bip = mne.io.RawArray(np.array(bip_ch_data), info)
    return bip


def read_seeg_xyz(subj_proc_dir):
    lines = []
    fname = os.path.join(subj_proc_dir, 'elec/seeg.xyz')
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            name, *sxyz = line.strip().split()
            xyz = [float(_) for _ in sxyz]
            lines.append((name, xyz))
    return lines

def gain_reorder(bip_gain,raw,bip_names):
    gain_pick = []
    raw_drop = []
    for i, ch_name in enumerate(raw.ch_names):
        if ch_name in bip_names:
            gain_pick.append(bip_names.index(ch_name))
        else:
            raw_drop.append(ch_name)
    raw = raw.drop_channels(raw_drop)
    gain_pick = np.array(gain_pick)
    picked_gain = bip_gain[gain_pick]
    
    return picked_gain, raw
            
def find_vhdrs(pid):
    
    raw_path = os.path.join('/', 'mnt','hgfs','EPINOV',pid)
    vhdr_pattern = os.path.join(raw_path, '*/ieeg/*seizure*.vhdr')
    vhdrs = glob.glob(vhdr_pattern)
    print('found VHDRs:')
    for vhdr in vhdrs:
        print(' - ' + vhdr)
    #print('removing second artifacted seizure')
    #del vhdrs[1]
    return vhdrs


def read_one_seeg(subj_proc_dir, vhdrname,remove_cerebellar=True): #mln
    raw = mne.io.read_raw_brainvision(vhdrname, preload=True)
    raw._data *= 1e6
    
    # read the bad channel
    fname_bad = f'{vhdrname}.bad'
    if op.exists(fname_bad):
        raw.load_bad_channels(fname_bad)
    else:
        print('>> Warning: bad channels file', fname_bad, 'not found => assuming no bad channels')

    # get rid of possible spaces in the names, which cause errors in the comparison with GARDEL names
    new_names = [name.replace(" ", "") for name in raw.ch_names]
    new_names = [name.replace(".", "") for name in new_names]
    raw.rename_channels(dict(zip(raw.ch_names, new_names)))

    # read from GARDEL file
    seeg_xyz = read_seeg_xyz(subj_proc_dir)
    seeg_xyz_names = [label for label, _ in seeg_xyz]

    # check wether all GARDEL channels exist in the raw seeg file
    # assert all([name in raw.ch_names for name in seeg_xyz_names]), "Not all GARDEL channel names are in the raw SEEG file."

    # check raw.ch_names against seeg_xyz_names to get rid of "MKR+" and similar channels, which are wrongly labeled as EEG and not discarded by the previous step
    raw = raw.pick_channels(seeg_xyz_names)
    raw = raw.pick_types(meg=False, eeg=True, exclude="bads")

    # read gain
    inv_gain_file = f'{subj_proc_dir}/elec/gain_inv-square.vep.txt'
    invgain = np.loadtxt(inv_gain_file)
    
    bip_gain_inv_minus, bip_xyz, bip_name = bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names)
    
    bip_gain_inv_prior, _, _ =  bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names, is_minus=False)
    # read the onset and offset
    fname_remark = f'{vhdrname}.mrk'
    with open(fname_remark) as fd:
        for line in fd.read().splitlines():
            if line.lower().startswith('seizure onset'):
                seizure_onset = line.split('\t')[2]
            if line.lower().startswith('seizure offset'):
                seizure_offset = line.split('\t')[2]

    bip = _bipify_raw(raw)
    gain,bip =gain_reorder(bip_gain_inv_minus,bip,bip_name)
    gain_prior, _ = gain_reorder(bip_gain_inv_prior,bip,bip_name)
    # remove the cerebellar
    roi = read_vep_mrtrix_lut()
    if remove_cerebellar:
        cereb_cortex = ['Left-Cerebellar-cortex','Right-Cerebellar-cortex']

        gain_prior.T[roi.index('Left-Cerebellar-cortex')] = gain_prior.T[-1]*0
        gain_prior.T[roi.index('Right-Cerebellar-cortex')] = gain_prior.T[-1]*0
        gain.T[roi.index('Left-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1]))*np.min(gain)
        gain.T[roi.index('Right-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1]))*np.min(gain)
    
    basicfilename=vhdrname.split('_ieeg.vhdr')[0]
    basicfilename=basicfilename.split('ieeg/')[1]
    seeg_info={}
    seeg_info['fname'] = f'{basicfilename}'
    seeg_info['onset'] = float(seizure_onset)
    seeg_info['offset'] = float(seizure_offset)
    #seeg_info['time_series'] = bip.get_data().T
    seeg_info['sfreq'] = bip.info['sfreq']
    #seeg_info['picks'] = picks

    return seeg_info, bip, gain, gain_prior

def bfilt(data, samp_rate, fs, mode, order=3, axis = -1):
    b, a = signal.butter(order, 2*fs/samp_rate, mode)
    return signal.filtfilt(b, a, data, axis)

def seeg_log_power(a, win_len, pad=True):
    envlp = np.empty_like(a)
    # pad with zeros at the end to compute moving average of same length as the signal itself
    envlp_pad = np.pad(a, ((0, win_len), (0, 0)), 'constant')
    for i in range(a.shape[0]):
        envlp[i, :] = np.log(np.mean(envlp_pad[i:i+win_len, :]**2, axis=0))
    return envlp

def compute_slp_sim(slp, sfreq, hpf=10.0, lpf=1.0, filter_order=5.0):

    #start_idx = int(seeg['onset'] * seeg['sfreq']) - int(seeg['sfreq'])
    #end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq'])
    #slp = seeg['time_series'][start_idx:end_idx]
    # Remove outliers i.e data > 2*sd
    for i in range(slp.shape[1]):
        ts = slp[:, i]
        ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
    # High pass filter the data
    slp = bfilt(slp, sfreq, hpf, 'highpass', axis=0)
    # Compute seeg log power
    slp = seeg_log_power(slp, 100)
    # Remove outliers i.e data > 2*sd
    for i in range(slp.shape[1]):
        ts = slp[:, i]
        ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
    # Low pass filter the data to smooth
    slp = bfilt(
        slp, sfreq, lpf, 'lowpass', axis=0)
    return slp

def prepare_data_feature_sim(SC, seeg, gain, sfreq, hpf=10.0, lpf=1.0):
  
       
    slp = compute_slp_sim(seeg, sfreq, hpf, lpf)
    data = {
        'SC': SC,
        'gain': gain,
        'slp': slp
    }
    return data



def prior_sim_slp(slp, sfreq,nperseg):
    Cs = []
    for F, T, C in _specn(slp, sfreq, nperseg):
        C[C<10e-15] = 10e-15
        Cs.append(np.log(C))
    Cs = np.array(Cs)
    return F, T, Cs

def compute_slp(seeg, bip, hpf=10.0, lpf=1.0, ts_base = 5,filter_order=5.0):
    
    base_length = int(seeg['sfreq']*ts_base)

    start_idx = int(seeg['onset'] * seeg['sfreq']) - base_length
    end_idx = int(seeg['offset'] * seeg['sfreq']) + base_length
    slp = bip.get_data().T[start_idx:end_idx]
    
    #start_idx = int(seeg['onset'] * seeg['sfreq']) - int(seeg['sfreq'])
    #end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq'])
    #slp = seeg['time_series'][start_idx:end_idx]
    # Remove outliers i.e data > 2*sd
    for i in range(slp.shape[1]):
        ts = slp[:, i]
        ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
    # High pass filter the data
    slp = bfilt(slp, seeg['sfreq'], hpf, 'highpass', axis=0)
    # Compute seeg log power
    slp = seeg_log_power(slp, 100)
    # Remove outliers i.e data > 2*sd
    for i in range(slp.shape[1]):
        ts = slp[:, i]
        ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
    # Low pass filter the data to smooth
    slp = bfilt(
        slp, seeg['sfreq'], lpf, 'lowpass', axis=0)
    return slp


def prepare_data_feature(data_dir, seeg_info, bip, gain, hpf=10.0, lpf=1.0,ts_base = 5):
    try:
        #with zipfile.ZipFile(
        #        f'{data_dir}/tvb/connectivity.vep.zip') as sczip:
        #    with sczip.open('weights.txt') as weights:
        SC = np.loadtxt(f'{data_dir}/dwi_correct_bvec/counts.vep.txt')
        #SC = np.loadtxt(f'{data_dir}/dwi/counts.vep.txt')
        SC[np.diag_indices(SC.shape[0])] = 0
        SC = SC/SC.max()
    except FileNotFoundError as err:
        print(f'{err}: Structural connectivity not found for {data_dir}')
        return
       
    slp = compute_slp(seeg_info, bip, hpf, lpf,ts_base)
    data = {
        'SC': SC,
        'gain': gain,
        'slp': slp
    }
    return data

def _spec1(y, sfreq, nperseg):
    return scipy.signal.spectrogram(y, sfreq, nperseg=nperseg)


def _specn(raw_data, sfreq, nperseg):
    args = [(y, sfreq, nperseg) for y in raw_data]
    with mp.Pool(12) as p:
        results = p.starmap(_spec1, args)
    return results


def prior_slp(bip,seeg, sfreq, nperseg):
    ts = 5
    base_length = int(seeg['sfreq']*ts)

    start_idx = int(seeg['onset'] * seeg['sfreq']) - base_length
    end_idx = int(seeg['offset'] * seeg['sfreq']) + 1
    slp = bip.get_data().T[start_idx:end_idx].T
    
    Cs = []
    for F, T, C in _specn(slp, sfreq, nperseg):
        C[C<10e-15] = 10e-15
        Cs.append(np.log(C))
    Cs = np.array(Cs)
    return F, T, Cs

def onset_delay(sz, threshperc=85):
    # sz.shape == (sensor, time)
    # threshold on 85 pct:
    m = sz > np.percentile(sz.flat[:], threshperc)
    # make mask false if less than 5 consecutive windows above threshold:
    m = np.array([binary_erosion(m_, np.r_[1, 1, 1, 1, 1]) for m_ in m])
    # find index of first window above threshold
    # j.shape == (n_sensor,)
    j = np.array([np.argwhere(_)[0, 0] + 5 if _.any() else len(_) for _ in m])
    # take reciprocal to have big value for early onset
    ji = (m.shape[1] + 5) / j
    # return value normalized 0 to 1
    return ji / ji.max()

def vep_gb(C,F,lp,hp,threshperc):
    C_ = (C * ((F > hp) * (F < lp))[:, None]).sum(axis=1)
    # Cp = C_ > np.percentile(C_.flat[:], 90)
    vep = onset_delay(C_,threshperc)
    return vep, C_

def plot_vep_priors(d0_prior, data, th_prior, roi = read_vep_mrtrix_lut(), title=None, figsize=(20, 10)):
    
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0)
    ax1.set_title(title)
    ax1.plot(d0_prior,"*")
    ax1.axhline(th_prior)
    #pl.xticks(np.r_[:len(roi)], roi, rotation=90);
    ns = 162
    barlist=ax2.bar(np.arange(0, ns), data['x0_mu']+3, color='black', alpha=0.3)
    plt.xticks(np.r_[:len(roi)], roi, rotation = 90, fontsize =9);
    
    #plt.title('A. $VEP_A$: delay on frequency powers', fontsize = 26);
    
    plt.xlim([-1,163])
    # ax2.set_yticklabels([item-3. for item in ax2.get_yticks()])
    ax2.set_yticklabels(['{:.2f}'.format(item-3.) for item in ax2.get_yticks()])
    plt.gcf().subplots_adjust(bottom=0.4)
    # plt.savefig(f'{results_dir}/figuresBT/{basicfilename}_3_prior_results_.png')

    return fig    


