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
import matplotlib
import subprocess
import json
options = ['vep_M', 'vep_W', 'vep_EI', 'vep_nullSC', 'vep_priornon']


def read_vep_mrtrix_lut():
    roi_names = []
    with open('/root/capsule/code/util/data/VepMrtrixLut.txt', 'r') as fd:
        for line in fd.readlines():
            i, roi_name, *_ = line.strip().split()
            roi_names.append(roi_name)
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

def prepare_data_feature_op(SC, seeg, gain, sfreq, hpf=10.0, lpf=1.0):
  
       
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

def plot_data_features_ax(data_features, plot_tt, ax, title=None, maxima_highlighted=None, onset_offset=None):

    ax.plot(plot_tt, data_features, color='black', alpha=0.3)

    if not maxima_highlighted is None:
        if not (isinstance(maxima_highlighted, dict)):
            print('>> WARNING: maxima_highlighted is expected to be a dict with keys = [''n'', ''ch_names''] and optionnal keys = [''fontsize'', ''legend_location'']')
        else:
            maxima = np.amax(data_features, axis=0)
            maxima_index_array = np.argsort(maxima)
            linewidth = 2
            cycler = matplotlib.rcParams['axes.prop_cycle']
            for c, i in enumerate(maxima_index_array[::-1][:maxima_highlighted['n']]):
                color = matplotlib.colors.hex2color(cycler.by_key()['color'][c % len(cycler.by_key()['color'])])
                label = maxima_highlighted['ch_names'][i]
                if c < len(cycler.by_key()['color']):
                    ax.plot(plot_tt, data_features[:, i], color=color, label=label, linewidth=linewidth )
                else:
                    ax.plot(plot_tt, data_features[:, i], color=color, label=label, linewidth=linewidth, linestyle='--')

            leg_ncol = maxima_highlighted['n'] // 5
            if maxima_highlighted['n'] % 5 > 0:
                leg_ncol += 1
            if 'legend_location' in maxima_highlighted:
                loc = maxima_highlighted['legend_location']
            else:
                loc = 'lower left'
            if 'fontsize' in maxima_highlighted:
                ax.legend(loc=loc, ncol=leg_ncol, fontsize=maxima_highlighted['fontsize'])
            else:
                ax.legend(loc=loc, ncol=leg_ncol)

    if onset_offset:
        for x in onset_offset:
            ax.axvline(x, color='DeepPink', lw=3)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('SLP', fontsize=12)
    ax.set_title(title, fontsize=16)


def plot_fitting_target(data, ch_names, pid, fname_suffix, maxima_highlighted=None, save_fname=None, close=False):
    
    fig = plt.figure(figsize=(30,15))

    ax = plt.subplot(211)
#     plt.plot(data['slp'], color='black', alpha=0.3)
#     plt.xlabel('Time', fontsize=12)
#     plt.ylabel('SLP', fontsize=12)
#     plt.title(f'{pid}: seizure{fname_suffix}: target feature', fontsize=16)
# #    plt.xlim([0,150])
    data_features = data['slp']
    title = f'{pid}: seizure{fname_suffix}: target feature'
    plot_data_features_ax(data_features, ax, title=title, maxima_highlighted=maxima_highlighted)

    plt.subplot(212)
    plt.bar(np.r_[1:data['ns']+1],data['snsr_pwr'], color='black', alpha=0.3)
    plt.xlabel('Electrodes', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.xticks(np.r_[1:len(ch_names)+1], ch_names,fontsize=11, rotation=90)
    plt.xlim([1,len(ch_names)+1])
    plt.title('SEEG channel power', fontweight='bold')

    if save_fname:
        print('>> Save', save_fname)
        plt.savefig(save_fname)
    if close:
        plt.close()

    return fig
def plot_sc(sc, pid, option=None, save_fname=None, close=False):
    roi = read_vep_mrtrix_lut()
    fig = plt.figure(figsize=(30,30))
    if option == 'vep_nullSC':
        plt.imshow(sc);
        plt.title(f'{pid}: Normalized SC',fontsize=12, fontweight='bold');
    else : 
        plt.imshow(sc,norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=sc.max()));
        plt.title(f'{pid}: Normalized SC (log scale)',fontsize=12, fontweight='bold');
    plt.xticks(np.r_[:len(roi)], roi, rotation = 90);
    plt.yticks(np.r_[:len(roi)], roi);
    if save_fname:
        print('>> Save', save_fname)
        plt.savefig(save_fname)
    if close:
        plt.close()
    return fig


def plot_gain_matrix(gain, ch_names, pid, option=None, save_fname=None, close=False):
    roi = read_vep_mrtrix_lut()
    fig = plt.figure(figsize=(30,30))
    plt.imshow(gain,norm=matplotlib.colors.LogNorm(vmin=gain.min(), vmax=gain.max()));
    plt.xticks(np.r_[:len(roi)], roi, rotation = 90);
    plt.yticks(np.r_[:len(ch_names)], ch_names);
    plt.xlabel('Region#', fontsize=12);
    plt.ylabel('Channel#', fontsize=12);
    plt.title(f'{pid}: Gain Matrix (log scale)',fontsize=12, fontweight='bold');
    plt.colorbar()
    if save_fname:
        print('>> Save', save_fname)
        plt.savefig(save_fname)
    if close:
        plt.close(fig)
    return fig

def save_data_feature_parameters_(dfp_fname, vhdr_fname, onset, offset, scaleplt, hpf, lpf, ts_on, ts_off, ts_cut, ts_off_cut, th_prior_W, th_prior_M, act_chs, replace_onset, replace_offset, replace_scale, bad_contacts, remove_roi):

    if op.exists(dfp_fname):
        fd = open(dfp_fname, 'r')
        parameters = json.load(fd)
        fd.close()
    else:
        os.makedirs(op.dirname(dfp_fname), exist_ok=True)
        parameters = {}

    parameters[vhdr_fname] = {'onset': onset, 'offset': offset, 'scaleplt': scaleplt, 'hpf': hpf, 'lpf': lpf,
                              'ts_on': ts_on, 'ts_off': ts_off, 'ts_cut': ts_cut, 'ts_off_cut': ts_off_cut,
                              'th_prior_W': th_prior_W, 'th_prior_M': th_prior_M, 'act_chs': act_chs,
                              'replace_onset': replace_onset, 'replace_offset': replace_offset,
                              'replace_scale': replace_scale,
                              'bad_contacts': bad_contacts, 'remove_roi': remove_roi}

    print('>> Save current data_feature__parameters to', dfp_fname)
    print('\tvhdr_fname:', vhdr_fname)
    print('\tseizure onset:', onset)
    print('\tseizure offset:', offset)
    print('\tbad_contacts:', bad_contacts)
    print('\tremove_roi:', remove_roi)
    print('\tscaleplt:', scaleplt)
    print('\thpf:', hpf)
    print('\tlpf:', lpf)
    print('\tts_on:', ts_on)
    print('\tts_off:', ts_off)
    print('\tts_cut:', ts_cut)
    print('\tth_prior_W:', th_prior_W)
    print('\tth_prior_M:', th_prior_M)
    print('\treplace_onset:', replace_onset)
    print('\treplace_offset:', replace_offset)
    print('\treplace_scale:', replace_scale)

    fd = open(dfp_fname, 'w')
    json.dump(parameters, fd)
    fd.close()

