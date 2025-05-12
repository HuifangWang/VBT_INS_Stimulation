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
import json
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.gridspec as gridspec
import stan
from itertools import compress
import matplotlib
import subprocess
sys.path.append('../util/')
import gain_matrix_seeg



def read_vep_mrtrix_lut():
    roi_names = []
    fname = op.join(op.dirname(op.dirname(op.realpath(__file__))), 'util/data/VepMrtrixLut.txt')
    with open(fname, 'r') as fd:
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

def obtain_elec_source_dis(basic_dir):

    
    tvb_zipfile = f'{basic_dir}/tvb/connectivity.vep.zip'
    sensors_file = f'{basic_dir}/elec/seeg.xyz'
    nregions = gain_matrix_seeg.get_nregions(tvb_zipfile)
    sensors_pos = np.genfromtxt(sensors_file, usecols=[1, 2, 3])
    surf_dir = f'{basic_dir}/tvb/'
    parcellation = 'vep'
    use_subcort = True
    verts, normals, areas, regmap = gain_matrix_seeg.read_surf(surf_dir, parcellation, use_subcort)
    vertices=verts
    region_mapping=regmap
    sensors = sensors_pos
    seeg_xyz = read_seeg_xyz(basic_dir)
    seeg_xyz_names = [label for label, _ in seeg_xyz]
    
    nverts = vertices.shape[0]
    nsens = sensors.shape[0]

    reg_map_mtx = np.zeros((nverts, nregions), dtype=int)
    for i, region in enumerate(region_mapping):
        if region >= 0:
            reg_map_mtx[i, region] = 1
    gain_mtx_vert = np.zeros((nsens, nverts))
    for sens_ind in range(nsens):
        a = sensors[sens_ind, :] - vertices
        na = np.sqrt(np.sum(a**2, axis=1))
        gain_mtx_vert[sens_ind, :] = 1 / na**2
        
    dis_elec_source = np.zeros((nsens,nregions))
    for indsens in np.arange(nsens):
        for ireg in np.arange(nregions):
            dis_elec_source[indsens,ireg] = np.max(gain_mtx_vert[indsens,np.where(reg_map_mtx[:,ireg] ==1)])
     
    return dis_elec_source

def generate_srtss_maps_re(subj_proc_dir, jsfname, remove_cerebellar=True,bad_contacts=False): #mln

    
    with open(f'{subj_proc_dir}/seeg/fif/{jsfname}.json', "r") as fd:
        js = json.load(fd)

    fifname = js['filename']
    raw = mne.io.Raw(f'{subj_proc_dir}/seeg/fif/{fifname}', preload=True)
    #drops = [_ for _ in (js["bad_channels"] + js["non_seeg_channels"]) if _ in raw.ch_names]
    
    
   
    #if bad_contacts:
        #print('>> Extending bad contacts with '+', '.join(bad_contacts))
    raw.info['bads'] = [] #.extend(bad_contacts)

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
    
    dis_elec_source = obtain_elec_source_dis(subj_proc_dir)
    bip_gain_inv_prior, _, _ =  bipolarize_gain_minus(dis_elec_source, seeg_xyz, seeg_xyz_names, is_minus=False)
    # read the onset and offset
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
    
    
    return bip.ch_names, gain_prior


def read_one_seeg_re(subj_proc_dir, jsfname,remove_cerebellar=True, bad_contacts=None): #mln
    with open(f'{subj_proc_dir}/seeg/fif/{jsfname}.json', "r") as fd:
        js = json.load(fd)

    fifname = js['filename']
    raw = mne.io.Raw(f'{subj_proc_dir}/seeg/fif/{fifname}', preload=True)
    drops = [_ for _ in (js["bad_channels"] + js["non_seeg_channels"]) if _ in raw.ch_names]
    
    
    
    if bad_contacts:
        print('>> Extending bad contacts with '+', '.join(bad_contacts))
        drops.extend(bad_contacts)
    raw = raw.drop_channels(drops)
    basicfilename=jsfname #.split('.json')[0]
    #basicfilename=basicfilename.split('/seeg/fif/')[1]
     # read gain
   
    seeg_xyz = read_seeg_xyz(subj_proc_dir)
    seeg_xyz_names = [label for label, _ in seeg_xyz]
    
    
    inv_gain_file = f'{subj_proc_dir}/elec/gain_inv-square.vep.txt'
    invgain = np.loadtxt(inv_gain_file)
    
    bip_gain_inv_minus, bip_xyz, bip_name = bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names)
    
    #dis_elec_source = obtain_elec_source_dis(subj_proc_dir)
    #bip_gain_inv_prior, _, _ =  bipolarize_gain_minus(dis_elec_source, seeg_xyz, seeg_xyz_names, is_minus=False)
        # read the onset and offset
    
    seizure_onset = js['onset']
            
    seizure_offset = js['termination']
                
    
    bip = _bipify_raw(raw)
    gain,bip =gain_reorder(bip_gain_inv_minus,bip,bip_name)
    #gain_prior, _ = gain_reorder(bip_gain_inv_prior,bip,bip_name)
    # remove the cerebellar from forward solution
    
    import pickle
    sstscfname = f"{subj_proc_dir}/tvb/sstsc_Mapping.vep.pickle"
    if not op.isfile(sstscfname):
        print('>> first time to generate the mapping files which may take longer time than usual')
        ch_names, prior_gain = generate_srtss_maps_re(subj_proc_dir,jsfname)
        maping_data_1 = {'ch_names':ch_names,'prior_Mapping':prior_gain}
        with open(sstscfname, 'wb') as fd:
            pickle.dump(maping_data_1, fd)
        
    with open(sstscfname, 'rb') as fdrb:
        maping_data = pickle.load(fdrb)
    
    bad_ch = []
    for ind_ch, ich in enumerate(maping_data['ch_names']):
        if ich not in bip.ch_names:
            bad_ch.append(ind_ch)
    gain_prior = np.delete(maping_data['prior_Mapping'], bad_ch, axis=0)
    
    roi = read_vep_mrtrix_lut()
    if remove_cerebellar:
        cereb_cortex = ['Left-Cerebellar-cortex','Right-Cerebellar-cortex']

        gain_prior.T[roi.index('Left-Cerebellar-cortex')] = gain_prior.T[-1]*0
        gain_prior.T[roi.index('Right-Cerebellar-cortex')] = gain_prior.T[-1]*0
        gain.T[roi.index('Left-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1]))*np.min(gain)
        gain.T[roi.index('Right-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1]))*np.min(gain)
    
    seeg_info={}
    seeg_info['fname'] = f'{basicfilename}'
    seeg_info['onset'] = float(seizure_onset)
    seeg_info['offset'] = float(seizure_offset)
    seeg_info['sfreq'] = bip.info['sfreq']


    return seeg_info, bip, gain, gain_prior


def get_channels_max_gain_sources(bip, gain):
    max_gain_sources = []
    roi = read_vep_mrtrix_lut()
    for c, _ in enumerate(bip.ch_names):
        max_gain_source = roi[np.argmax(gain[c])]
        max_gain_sources.append(max_gain_source)
    return max_gain_sources


def plot_sensor_data(bip, gain, seeg_info, ts_on, ts_cut=None, data_scaleplt=1, datafeature=None, datafeature_scaleplt=0.7, title=None, figsize=[40,70], yticks_fontsize=26, ch_selection=None):
    
    # plot real data at sensor level
    show_ch = bip.ch_names
    nch_source = []
    roi = read_vep_mrtrix_lut()
    for ind_ch, ichan in enumerate(show_ch):
        isource = roi[np.argmax(gain[ind_ch])]
        nch_source.append(f'{isource}:{ichan}')

    # plot ts_on sec before and after
    base_length = int(ts_on * seeg_info['sfreq'])
    start_idx = int(seeg_info['onset'] * seeg_info['sfreq']) - base_length
    end_idx = int(seeg_info['offset'] * seeg_info['sfreq']) + base_length

    y = bip.get_data()[:,start_idx:end_idx]
    t = bip.times[start_idx:end_idx]

    # do same clipping as for datafeature in prepare_data_feature when plotting both
    if datafeature is not None and ts_cut is not None: 
        cut_off_N = int(ts_cut*seeg_info['sfreq'])
        y = y[:,cut_off_N:-1]
        t = t[cut_off_N:-1]

    f = plt.figure(figsize=figsize)

    if ch_selection is None:
        ch_selection = range(len(bip.ch_names))
    for ch_offset, ch_ind in enumerate(ch_selection):
        plt.plot(t, data_scaleplt*y[ch_ind]+ch_offset, 'blue', lw=0.5)
        if datafeature is not None:
            plt.plot(t, datafeature_scaleplt*(datafeature.T[ch_ind]-datafeature[0,ch_ind])+ch_offset,'red',lw=1.5)
            
    vlines=[seeg_info['onset'], seeg_info['offset']]
    for x in vlines:
        plt.axvline(x, color='DeepPink', lw=3)

    # annotations
    if bip.annotations:
        for ann in bip.annotations:
            descr = ann['description']
            start = ann['onset']
            end = ann['onset'] + ann['duration']
            # print("'{}' goes from {} to {}".format(descr, start, end))
            if descr == 'seeg_bad_segments':
                plt.axvline(start, color='red', lw=1)
                plt.axvline(end, color='red', lw=1)

    plt.xticks(fontsize=18)
    plt.xlim([t[0],t[-1]])
    plt.yticks(range(len(ch_selection)), [nch_source[ch_index] for ch_index in ch_selection], fontsize=yticks_fontsize)
    plt.ylim([-1,len(ch_selection)+0.5])
    
    plt.title(title,fontsize = 16)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(top=0.97)
    plt.tight_layout()
    
    return f


def plot_data_features(data_features, plot_tt, title=None, figsize=(30,15), maxima_highlighted=None, onset_offset=None):
    fig, ax = plt.subplots(figsize=figsize)
    plot_data_features_ax(data_features,plot_tt, ax, title=title, maxima_highlighted=maxima_highlighted, onset_offset=onset_offset)
    return fig


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


def compute_slp(seeg, bip, hpf=10.0, lpf=1.0, ts_base = 5, ts_off = 5 ,filter_order=5.0):
    base_length = int(seeg['sfreq']*ts_base)

    start_idx = int(seeg['onset'] * seeg['sfreq']) - base_length
    end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq']*ts_off)
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


def data_feature_parameters_fname(pid, data_base_dir='/data/retrospective'):
    return op.join(data_base_dir, 'vep_Nov20', pid, 'vep_dfs', 'data_feature_parameters.txt' )


def save_data_feature_parameters(pid, vhdr_fname, onset, offset, scaleplt, hpf, lpf, ts_on, ts_off, ts_cut, th_prior_W, th_prior_M,act_chs=[], bad_contacts=[],removebaseline=True):

    fname = data_feature_parameters_fname(pid)
    if op.exists(fname):
        fd = open(fname, 'r')
        parameters = json.load(fd)
        fd.close()
    else:
        os.makedirs(op.dirname(fname), exist_ok=True)
        parameters = {}
        

    parameters[vhdr_fname] = {'onset': onset, 'offset': offset, 'scaleplt': scaleplt, 'hpf': hpf, 'lpf': lpf, 'ts_on': ts_on, 'ts_off':ts_off, 'ts_cut': ts_cut,'th_prior_W': th_prior_W, 'th_prior_M': th_prior_M, 'act_chs':act_chs, 'bad_contacts': bad_contacts,'removebaseline':removebaseline}
    
    print('>> Save current data_feature__parameters to', fname)
    print('\tvhdr_fname:', vhdr_fname)
    print('\tseizure onset:', onset)
    print('\tseizure offset:', offset)
    print('\tbad_contacts:', bad_contacts)
    print('\tscaleplt:', scaleplt)
    print('\thpf:', hpf)
    print('\tlpf:', lpf)
    print('\tts_on:', ts_on)
    print('\tts_off:', ts_off)
    print('\tts_cut:', ts_cut)
    print('\tth_prior_W:', th_prior_W)
    print('\tth_prior_M:', th_prior_M)


    fd = open(fname, 'w')
    json.dump(parameters, fd)
    fd.close()
    
    
def load_data_feature_parameters(pid, vep_fit_id=None, vhdr_fname=None, data_base_dir='/data/retrospective'):
    fname = data_feature_parameters_fname(pid, data_base_dir=data_base_dir)
    fd = open(fname, 'r')
    parameters = json.load(fd)
    fd.close()
    if not vhdr_fname in parameters.keys():
        raise Exception(f'\n{vhdr_fname} is not a valid key. Valid keys are:\n'+'\n'.join(parameters.keys()))
    return parameters[vhdr_fname]


def load_sc(data_dir):
    try:
        with zipfile.ZipFile(
                f'{data_dir}/tvb/connectivity.vep.zip') as sczip:
            with sczip.open('weights.txt') as weights:
                SC = np.loadtxt(weights)
                SC[np.diag_indices(SC.shape[0])] = 0
                SC = SC/SC.max()
    except FileNotFoundError as err:
        print(f'{err}: Structural connectivity not found for {data_dir}')
   
    return SC


def prepare_data_feature(data_dir, seeg_info, bip, gain, hpf=10.0, lpf=1.0, ts_base=5, ts_off=None,ts_cut =None,removebaseline=True):
    #ts_cut = ts_base/4
    try:
        SC = load_sc(data_dir)
    except FileNotFoundError as err:
        print(f'{err}: Structural connectivity not found for {data_dir}')
        return
    
    if ts_off is None:
        ts_off = ts_base
        slp = compute_slp(seeg_info, bip, hpf, lpf,ts_base,ts_off)
    cut_off_N = int(ts_cut*seeg_info['sfreq'])
    slp = slp[cut_off_N:-1,:]
    if removebaseline:
        baseline_N = int((ts_base-ts_cut)*seeg_info['sfreq'])
        slp = slp - np.mean(slp[:baseline_N,:],axis=0 )

    
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


def prior_slp(bip, seeg, sfreq, nperseg):
    ts = 1
    NWindow=200
    base_length = int(seeg['sfreq']*ts)

    start_idx = int(seeg['onset'] * seeg['sfreq']) - base_length
    end_idx = int(seeg['offset'] * seeg['sfreq']) + 1
    slp = bip.get_data().T[start_idx:end_idx].T
    
    Cs = []
    Seizured = seeg['offset'] - seeg['onset']
    if  Seizured<= NWindow:
        noverlap = np.ceil((1-Seizured/NWindow)*nperseg)
        [FV, TV, CV] = scipy.signal.spectrogram(slp, sfreq, nperseg=nperseg, noverlap=noverlap)
    else:
        [FV, TV, CV] = scipy.signal.spectrogram(slp, sfreq, nperseg=nperseg)   
    # for F, T, C in zip(FV, TV, CV):
    #     C[C<10e-15] = 10e-15
    #     Cs.append(np.log(C))
    # Cs = np.array(Cs)
    # return FV, TV, Cs
    
    CV[CV < 10e-15] = 10e-15
    
    CV = np.log(CV)
    
    return FV, TV, CV


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


def vep_gb(C, F, lp, hp, threshperc):
    C_ = (C * ((F > hp) * (F < lp))[:, None]).sum(axis=1)
    # Cp = C_ > np.percentile(C_.flat[:], 90)
    vep = onset_delay(C_,threshperc)
    return vep, C_


def results_dir_fun(pid, option, subj_proc_dir):
    results_dir = f'{subj_proc_dir}/{option}'
    return results_dir


def createResultsDirs(pid, option, subj_proc_dir):
    results_dir = results_dir_fun(pid, option, subj_proc_dir)
    os.makedirs(f'{results_dir}', exist_ok=True)
    os.makedirs(f'{results_dir}/logs', exist_ok=True)
    os.makedirs(f'{results_dir}/figuresBT', exist_ok=True)
    os.makedirs(f'{results_dir}/EZdelay', exist_ok=True)
    os.makedirs(f'{results_dir}/Rfiles', exist_ok=True)
    os.makedirs(f'{results_dir}/Cfiles', exist_ok=True)
    os.makedirs(f'{results_dir}/RfilesBT', exist_ok=True)
    os.makedirs(f'{results_dir}/OptimalBT', exist_ok=True)
    return results_dir
    

def compute_vep_priors(C, Freq_seeg, ch_names, option=None, cel_gain=None, title=None, figsize=None):
    
    if figsize:
        fig = pl.figure(figsize=figsize)
    else:
        if option is None:  # sensor level
            fig = pl.figure(figsize=[20,35])
        else:               # region level
            fig = pl.figure(figsize=[20,25])
        
    lowF = np.arange(10,90,10)
    NlowF = len(lowF)
    gs = gridspec.GridSpec(1, NlowF)
    threshperc = 90
    all_fb_d0 = []
    for indlowF, ilowF in enumerate(lowF):
        highF = np.arange(ilowF+10, 120, 10)
        all_Vep = []
        for ihighF in highF:
            sfb_vep, sfb_C_ = vep_gb(C, Freq_seeg, ihighF, ilowF, threshperc)
            if option is None:
                all_fb_d0.append(sfb_vep)
                all_Vep.append(sfb_vep)
            else:
                if option in ['vep_W']:
                    vep_smax = sfb_vep @ cel_gain
                elif option == 'vep_M':
                    vep_smax = np.zeros(cel_gain.shape[1])
                    for isfb, indroi_iv in zip(sfb_vep,np.argmax(cel_gain,axis=1)):
                        if vep_smax[indroi_iv] < isfb:
                            vep_smax[indroi_iv] = isfb
                else:
                    raise Exception('Invalid option: '+option) 
                all_fb_d0.append(vep_smax)
                all_Vep.append(vep_smax)
            
        all_Vep = np.array(all_Vep)
        
        ax = pl.subplot(gs[0, indlowF])
        plt.imshow(all_Vep.T[::-1], aspect='auto')
        pl.xticks(np.r_[:len(highF)], highF, rotation=90);
        pl.title(f'LF: {ilowF}', fontsize=14)
        if indlowF < 1:
            pl.yticks(np.r_[:len(ch_names)], ch_names[::-1]);
        else:
            pl.yticks([])
    if title:
        fig.suptitle(title)
    pl.colorbar()
    
    all_fb_d0 = np.array(all_fb_d0)
    #np.save(f'{results_dir}/EZdelay/ez_prior_{basicfilename}.npy',all_fb_d0)
    # np.save(f'{results_dir}/EZdelay/ez_prior_{basicfilename}.npy',all_fb_d0)
    
    #os.makedirs(f'{results_dir}/EZdelay', exist_ok=True)
    #np.save(f'{results_dir}/EZdelay/ez_prior_{basicfilename}_elec.npy',all_fb_d0_elec)
    return (fig, all_fb_d0)


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





def get_all_seizures(pid, jsonfile, verbose=True):
    """
    jsonfile - Path to a json file containing all the SEEG seizure recording file names.
    """
    fileList_pid = []
    print(pid)
    for jsonfile_i in jsonfile:
            with open(jsonfile_i,'r') as f:
                post_all = json.load(f)

            if post_all['type'] == 'Spontaneous seizure':

                fileList_pid.append(post_all['filename'].split('.raw.fif')[0])
        
    return fileList_pid


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


options = ['vep_M', 'vep_W', 'vep_EI', 'vep_nullSC', 'vep_priornon']


def generate_r_files(pid, n_szr, option, subj_proc_dir, EI_jsonfile,  remove_previous_r_files=True):

    assert option in options
    
    # print all seizures for this patient
    all_seizures = get_all_seizures(pid, verbose=True)
    
    results_dir = results_dir_fun(pid, option)
    createResultsDirs(pid, option, subj_proc_dir, subj_proc_dir)
    
    basicfilename = all_seizures[n_szr]
     
    if remove_previous_r_files:
        previous_r_files = []
        for d in ['Rfiles', 'RfilesBT']:
            pattern = op.join(f'{results_dir}', d, f'fit_data_{basicfilename}_hpf*_lpf*.R')
#            print('pattern', pattern)
            previous_r_files.extend(glob.glob(pattern))
            if d == 'Rfiles':
                pattern = op.join(f'{results_dir}', d, f'param_init_{basicfilename}_hpf*_lpf*.R')
#                print('pattern', pattern)
                previous_r_files.extend(glob.glob(pattern))
        #print(previous_r_files)
        print('>> First remove', len(previous_r_files), 'previous R files')
        for f in previous_r_files:
            os.remove(f)
    
    
    #    stan_fname = 'szr_prpgtn'
        
    # load previously prepared parameters
    parameters = load_data_feature_parameters(pid, basicfilename)
    onset = parameters['onset']
    offset = parameters['offset']
    hpf = parameters['hpf']
    lpf = parameters['lpf']
    ts_on = parameters['ts_on']
    ts_cut = parameters['ts_cut']
    ts_off = parameters['ts_off']
    bad_contacts = parameters['bad_contacts']
    if 'removebaseline' in parameters.keys():
        removebaseline=parameters['removebaseline']
    else:
        removebaseline=True
    
    fname_suffix = f'{basicfilename}_hpf{hpf}_lpf{lpf}'
    
    
    basic_option = 'vep_M'
    dir_basic_readR = results_dir_fun(pid, basic_option, subj_proc_dir)
    print('vep_M=',dir_basic_readR)
    if option == 'vep_M':
            
        th_prior = parameters['th_prior_M']
    
        # load priors
        all_fb_d0 = np.load(f'{results_dir}/EZdelay/ez_prior_{basicfilename}.npy')
        d0_prior = np.mean(all_fb_d0, axis=0)
        d0_prior = d0_prior/d0_prior.max()
        ez_prior = np.where(d0_prior>th_prior)
    
        
    elif option == 'vep_W':
        
        th_prior = parameters['th_prior_W']
    
        # load priors
        all_fb_d0 = np.load(f'{results_dir}/EZdelay/ez_prior_{basicfilename}.npy')
        d0_prior = np.mean(all_fb_d0, axis=0)
        d0_prior = d0_prior/d0_prior.max()
        ez_prior = np.where(d0_prior>th_prior)
    
    
    elif option == 'vep_EI':
        
        # load priors
        pid_ = pid
        import json

        with open (EI_jsonfile,'r') as ch_file:
            ch = json.load(ch_file)

        
        ez_prior = []
        roi = read_vep_mrtrix_lut()
        for ind_roi in ch[pid]['ez']:
            ez_prior.append(roi.index(ind_roi))
        ez_prior=np.array(ez_prior)            
    
    
    
    if option is basic_option:
        
        # get data features
        seeg_info, bip, gain, gain_prior = read_one_seeg_re(subj_proc_dir, basicfilename,bad_contacts=bad_contacts)
        seeg_info['onset'] = onset
        seeg_info['offset'] = offset
    
        data = prepare_data_feature(subj_proc_dir, seeg_info, bip, gain, hpf, lpf, ts_on,ts_off,ts_cut,removebaseline)
        ds_freq = int(data['slp'].shape[0]/150)
        data['slp'] = data['slp'][0:-1:ds_freq]
        data['snsr_pwr'] = (data['slp']**2).mean(axis=0)
        data['ns'], data['nn'] = data['gain'].shape
        data['nt'] = data['slp'].shape[0]
        data['x0_mu'] = -3.0*np.ones(data['nn'])
        data['x0_mu'][ez_prior] = -1.5
        
        
        # write R files
        x0 = data['x0_mu']
        amplitude = 1.0 
        offset = 0
        K = 1.0
        tau0 = 20
        eps_slp = 1.0
        eps_snsr_pwr = 1.0
        x_init = -2.0*np.ones(data['nn'])
        z_init = 3.5*np.ones(data['nn'])
        
        param_init = {'x0':x0, 'amplitude':amplitude,
                      'offset':offset, 'K':K, 'tau0':tau0, 'x_init':x_init, 'z_init':z_init,
                      'eps_slp':eps_slp, 'eps_snsr_pwr':eps_snsr_pwr}
        
        fname = f'{results_dir}/Rfiles/param_init_{fname_suffix}.R'
#        print('>> Save', fname)
        stan.rdump(fname, param_init)
        
        fname = f'{results_dir}/Rfiles/fit_data_{fname_suffix}.R'
#        print('>> Save', fname)
        stan.rdump(fname, data)
        
        
        # save 100 bootsraped R files
        ii = np.random.randint(0, data['ns']-1, 100)
        fname = f'{results_dir}/RfilesBT/fit_data_{fname_suffix}_[0-{len(ii)-1}].R'
#        print('>> Save 100 bootsraped R files in', fname)
        for id_bt, ibt in enumerate(ii):
            idata = data.copy()
            idata['slp'] = np.delete(idata['slp'], ibt, 1)
            idata['snsr_pwr'] = np.delete(idata['snsr_pwr'], ibt)
            idata['gain'] = np.delete(idata['gain'], ibt, 0)
            idata['ns'], idata['nn'] = idata['gain'].shape
        
            input_Rfile = f'fit_data_{fname_suffix}_{id_bt}.R'
            stan.rdump(f'{results_dir}/RfilesBT/{input_Rfile}', idata)
        
    else:
    
        # read and write the R files
        data = stan.rload(f'{dir_basic_readR}/Rfiles/fit_data_{fname_suffix}.R')
        odata = data.copy()
        odata['x0_mu'] = -3.0*np.ones(odata['nn'])
        if option in ['vep_W', 'vep_EI']:
            odata['x0_mu'][ez_prior] = -1.5
        elif option == "vep_nullSC" : # set all SC weights to 0 and use uninformative EZ prior
            odata['SC'] = np.zeros(np.shape(odata['SC']))
        elif option == "vep_priornon" : # use uninformative EZ prior
            pass
        
        # update the information file
        param_init = {'x0':           odata['x0_mu'], 
                      'amplitude':    1.0,
                      'offset':       0, 
                      'K':            1.0, 
                      'tau0':         20, 
                      'x_init':       -2.0*np.ones(data['nn']), 
                      'z_init':       3.5*np.ones(data['nn']),
                      'eps_slp':      1.0, 
                      'eps_snsr_pwr': 1.0}
    
        fname = f'{results_dir}/Rfiles/param_init_{fname_suffix}.R'
#        print('>> Save', fname)
        stan.rdump(fname, param_init)  
        
        fname = f'{results_dir}/Rfiles/fit_data_{fname_suffix}.R'
#        print('>> Save', fname)
        stan.rdump(fname, odata)
        
        
        # save 100 bootsraped R files
        # ffname = f'{results_dir}/RfilesBT/fit_data_{fname_suffix}_[0-99].R'
#        print('>> Save 100 bootsraped R files in', fname)
        for id_bt in np.arange(100):
            bt_input_Rfile = f'fit_data_{fname_suffix}_{id_bt}.R'
            bt_data = stan.rload(f'{dir_basic_readR}/RfilesBT/{bt_input_Rfile}')
            odata = bt_data.copy()
            odata['x0_mu'] = -3.0*np.ones(odata['nn'])
            if option in ['vep_W', 'vep_EI']:
                odata['x0_mu'][ez_prior] = -1.5
            elif option == "vep_nullSC" : # set all SC weights to 0 and use uninformative EZ prior
                odata['SC'] = np.zeros(np.shape(odata['SC']))
            elif option == "vep_priornon" : # use uninformative EZ prior
                pass
    
            stan.rdump(f'{results_dir}/RfilesBT/{bt_input_Rfile}', odata)
        

def fitting_optimization_errors(pid, seizure_ind=None, opts=options, n_bt=100):
    all_seizures = get_all_seizures(pid, verbose=False)
    seizures = [all_seizures[ind] for ind in seizure_ind] if seizure_ind else all_seizures
    for vhdr_fname in seizures:
        print('>> Optimization errors for', op.basename(vhdr_fname))
        for option in opts:
            (error_complete, errors_bt) = fitting_optimization_(pid, vhdr_fname, option)
            error_msg = '\t' + f'{option:<16}'
            error_msg += f'{error_complete}/1' + '\t'
            if error_complete > 0:
                error_msg += f'*'
            else:
                error_msg += f' '
            error_msg += f'{errors_bt:>8}/{n_bt}' + '\t'
            if errors_bt > 0:
                error_msg += f'*'
            else:
                error_msg += f' '
            print(error_msg)


def fitting_optimization_(pid, vhdr_fname, option, subj_proc_dir, n_bt=100,):
    optimization_error_msg = 'Optimization terminated with error: \n'

    pid_parameters = load_data_feature_parameters(pid, op.basename(vhdr_fname))
    hpf = pid_parameters['hpf']
    lpf = pid_parameters['lpf']

    basicfilename = op.basename(vhdr_fname).replace('_ieeg.vhdr', '')
    # print(basicfilename)
    fname_suffix = f'{basicfilename}_hpf{hpf}_lpf{lpf}'

    # complete fitting
    fname = op.join(results_dir_fun(pid, option, subj_proc_dir), 'logs', 'snsrfit_ode_' + fname_suffix + '.log')
    with open(fname, 'r') as fd:
        optimization_error = optimization_error_msg in fd.readlines()
    # print('\tmain fitting optimization error:', str(np.where(optimization_error)[0].size)+'/1')

    # bt fitting
    optimization_errors = np.full(n_bt, True, dtype=np.bool)
    for n in range(n_bt):
        fname = op.join(results_dir_fun(pid, option, subj_proc_dir), 'logs', 'snsrfit_ode_' + fname_suffix + f'_{n}.log')
        with open(fname, 'r') as fd:
            optimization_errors[n] = optimization_error_msg in fd.readlines()
    # print('\tbt fitting optimization error:', str(np.where(optimization_errors)[0].size)+f'/{n_bt}')
    return (np.where(optimization_error)[0].size, np.where(optimization_errors)[0].size)


def fit_patient(pid, subj_proc_dir, opts=options):
    stan_fname = 'szr_prpgtn'

    for opt in opts:
        print("processing option : " + opt)
        results_dir = f'{subj_proc_dir}/{opt}'

        process = subprocess.run(['bash', 'fit_seizure.sh', stan_fname, results_dir],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        print(process.stdout)

        process = subprocess.run(['bash', 'fit_seizure_BT.sh', stan_fname, results_dir],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        print(process.stdout)
        
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

