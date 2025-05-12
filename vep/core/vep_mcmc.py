import sys
import os.path as op

import vep_prepare_op
import numpy as np
import scipy
from scipy import linalg
import os
import matplotlib
import matplotlib.pyplot as plt
try:
    matplotlib.use('Qt5Agg')
except:
    pass
plt.interactive(True)
import stan
import analyze_fit
import subprocess
import pandas as pd
from parfor import parfor
import random


#stanmodel = 'rpne_039'
def specify_stan_model(stanmodel):
    rpne_039__stan_fname = op.join(op.dirname(op.realpath(__file__)), '../stan/', stanmodel)
    assert op.exists(rpne_039__stan_fname), f'Missing stan model: {rpne_039__stan_fname}'
    return rpne_039__stan_fname



def run_jobs_parallel(parallel_jobs_fname, n_jobs):
    command = ['time', '/usr/bin/parallel', '-j', f'{n_jobs}']
    fd = open(parallel_jobs_fname, 'r')
    process = subprocess.run(command,
                             stdin=fd,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    fd.close()
    print(process.stdout)


def mcmc__dname(subj_proc_dir, mcmc_fit_id):
    if not mcmc_fit_id:
        dname = op.join(subj_proc_dir, 'mcmc',)
    else:
        dname = op.join(subj_proc_dir, 'mcmc', mcmc_fit_id)
    return dname


def datafeatures__rpne_039__fname(vhdr_fname, mcmc_fit_id, subj_proc_dir, hpf, lpf):
    bname = op.basename(vhdr_fname).rstrip('_ieeg.vhdr')
    bname += f'_hpf{hpf}_lpf{lpf}__datafeatures.R'
    fname = op.join(mcmc__dname(subj_proc_dir=subj_proc_dir, mcmc_fit_id=mcmc_fit_id), 'vep_dfs', bname)
    return fname


def create__jobs__optimize__rpne_039(datafeatures_fname,
                                     stanmodel,
                                     init_uniform_distribution_range=5,
                                     n_optimize=50, n_jobs=20, tol_param=1e-10):

    datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
    output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'optimize_'+stanmodel, datafeatures_id)
    jobs_dir = op.join(output_dir, 'jobs')
    logs_dir = op.join(output_dir, 'logs')
    csv_dir = op.join(output_dir, 'csv')
    for d in [jobs_dir, logs_dir, csv_dir]:
        os.makedirs(d, exist_ok=True)

    parallel_jobs_fname = op.join(jobs_dir, f'do_all_local.bash')
    # print(f'>> Save list of jobs in {parallel_jobs_fname}')
    fd_parallel_jobs = open(parallel_jobs_fname, 'w')
    parallel_command = f'time /usr/bin/parallel -j {n_jobs} < {parallel_jobs_fname}'
    fd_parallel_jobs.write(f'# {parallel_command}\n')
    rpne_039__stan_fname = specify_stan_model(stanmodel) 
    for j in range(n_optimize):
        job_id = f'{datafeatures_id}__j{j}'
        job_fname = op.join(jobs_dir, f'{job_id}.bash')
        log_fname = op.join(logs_dir, f'{job_id}.log')
        # err_fname = op.join(logs_dir, f'{job_id}.err')
        output_file = op.join(csv_dir, f'{job_id}.csv')

        job = f"""{rpne_039__stan_fname} optimize algorithm=lbfgs tol_param={tol_param} iter=20000 save_iterations=0 \\\n"""\
              f"""data file={datafeatures_fname} \\\n"""\
              f"""init={init_uniform_distribution_range} \\\n"""\
              f"""output file={output_file} \\\n"""\
              f"""refresh=10"""

        fd = open(job_fname, 'w')
        fd.write('#!/bin/bash\n\n')
        fd.write(job)
        fd.close()
        # fd_parallel_jobs.write(f'bash {job_fname} > {log_fname} 2> {err_fname}\n')
        fd_parallel_jobs.write(f'bash {job_fname} >& {log_fname}\n')

    fd_parallel_jobs.close()
    #print()
    print(f'>> Save {j+1} jobs in {parallel_jobs_fname}')
    print(f'>> Run: {parallel_command}')

    return parallel_jobs_fname


def create__init_files__sample__rpne_039(datafeatures_fname, stanmodel, variables_of_interest=['lp__'],
                                         n_optimize=50, n_best=8):

    # read csv files and get the 'lp__'
    datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
    output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'optimize_'+stanmodel, datafeatures_id)
    logs_dir = op.join(output_dir, 'logs')
    csv_dir = op.join(output_dir, 'csv')

    lp = np.full(n_optimize, np.nan)
    j_ok, j_nok, j_exc = 0, 0, 0

    for j in range(n_optimize):
        job_id = f'{datafeatures_id}__j{j}'
        log_fname = op.join(logs_dir, f'{job_id}.log')
        print(f'Reading status from {log_fname}... ', end='', flush=True)
        try:
            status = analyze_fit.analyze_log_file(log_fname)
        except Exception as e:
            print(e)
            j_exc += 1
            continue
        print(status)
        if status == 'ok':
            j_ok += 1
            csv_fname = op.join(csv_dir, f'{job_id}.csv')
            assert op.exists(csv_fname)
            print(f'\t\t>> Reading lp__ from {csv_fname}...', end=' ', flush=True)
            samples = stan.read_samples([csv_fname], variables_of_interest=['lp__'])
            print('ok')
            lp[j] = samples['lp__'][0]
        else:
            j_nok += 1

    assert j_ok + j_nok + j_exc == n_optimize
    print()
    print(f'>> Jobs: {n_optimize}, ok: {j_ok}, nok: {j_nok}, exc: {j_exc}')
    # HW: there is bug for trial patient, don't forget to change.
    sorted_indicies = np.argsort(-lp)
    n_best_indices = sorted_indicies[:n_best]

    print()
    print(f'>> Keeping jobs {n_best_indices} with lp__ {lp[n_best_indices]}')
    print()


    # create init files with n_best estimations
    init_dir = op.join(output_dir, 'csv', 'init')
    os.makedirs(init_dir, exist_ok=True)
    for j in n_best_indices:
        job_id = f'{datafeatures_id}__j{j}'
        csv_fname = op.join(csv_dir, f'{job_id}.csv')
        print(f'Reading variables_of_interest from {csv_fname} ...', end=' ', flush=True)
        samples = stan.read_samples([csv_fname], variables_of_interest=variables_of_interest)
        print('ok')
        param_init = dict.fromkeys(variables_of_interest)
        for k in variables_of_interest:
            values = samples[k].squeeze()
            param_init[k] = values
        init_fname = op.join(init_dir, op.basename(csv_fname).rstrip('.csv')+'_init.R')
        print(f'>> Save {init_fname}')
        stan.rdump(init_fname, param_init)


def create__jobs__sample__rpne_039(datafeatures_fname,stanmodel, n_init=8, n_chains_per_init=2, n_jobs=None):
    rpne_039__stan_fname = specify_stan_model(stanmodel) 
    datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
    optimize_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'optimize_'+stanmodel, datafeatures_id)
    init_dir = op.join(optimize_output_dir, 'csv', 'init')
    init_files = os.listdir(init_dir)
    assert len(init_files) == n_init
    init_files = [op.join(init_dir, f) for f in init_files]

    sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample_'+stanmodel, datafeatures_id)
    jobs_dir = op.join(sample_output_dir, 'jobs')
    logs_dir = op.join(sample_output_dir, 'logs')
    csv_dir = op.join(sample_output_dir, 'csv')
    for d in [jobs_dir, logs_dir, csv_dir]:
        os.makedirs(d, exist_ok=True)

    if n_jobs is None:
        n_jobs = n_init*n_chains_per_init

    parallel_jobs_fname = op.join(jobs_dir, f'do_all_local.bash')
    # print(f'>> Save list of jobs in {parallel_jobs_fname}')
    fd_parallel_jobs = open(parallel_jobs_fname, 'w')
    parallel_command = f'time /usr/bin/parallel -j {n_jobs} < {parallel_jobs_fname}'
    fd_parallel_jobs.write(f'# {parallel_command}\n')

    num_warmup = 500
    num_samples = 500
    delta = 0.99
    max_depth = 7

    j = 0
    for init_file in init_files:
        for c in range(n_chains_per_init):
            job_id = f'{datafeatures_id}__j{j}'
            job_fname = op.join(jobs_dir, f'{job_id}.bash')
            log_fname = op.join(logs_dir, f'{job_id}.log')
            # err_fname = op.join(logs_dir, f'{job_id}.err')
            output_file = op.join(csv_dir, f'{job_id}.csv')
            output_diagnostic_file = op.join(csv_dir, f'{job_id}_diagnostic.csv')

            job = f"""# run one chain\n""" \
                  f"""\n""" \
                  f"""{rpne_039__stan_fname} id={c+1} \\\n""" \
                  f"""  sample \\\n""" \
                  f"""  save_warmup=0 num_warmup={num_warmup} num_samples={num_samples} \\\n""" \
                  f"""  adapt delta={delta} \\\n""" \
                  f"""  algorithm=hmc engine=nuts max_depth={max_depth} \\\n""" \
                  f"""  random seed=12345 \\\n""" \
                  f"""  data file={datafeatures_fname} \\\n""" \
                  f"""  init={init_file} \\\n""" \
                  f"""  output file={output_file} \\\n""" \
                  f"""      diagnostic_file={output_diagnostic_file} \\\n""" \
                  f"""  refresh=1\n"""

            fd = open(job_fname, 'w')
            fd.write('#!/bin/bash\n\n')
            fd.write(job)
            fd.close()
            # fd_parallel_jobs.write(f'bash {job_fname} > {log_fname} 2> {err_fname}\n')
            fd_parallel_jobs.write(f'bash {job_fname} >& {log_fname}\n')

            j += 1

    fd_parallel_jobs.close()
    print()
    print(f'>> Save {j} jobs in {parallel_jobs_fname}')
    print(f'>> Run : {parallel_command}')

    return parallel_jobs_fname


def chains_diagnostic(datafeatures_fname, stanmodel, n_init=8, n_chains_per_init=2):

    datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
    
    sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample_'+stanmodel, datafeatures_id)
    
    csv_dir = op.join(sample_output_dir, 'csv')
    fig_dir = op.join(csv_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    n_chains = n_init * n_chains_per_init
    chains_status = np.full((n_chains), True)

    for j in range(n_chains):
        job_id = f'{datafeatures_id}__j{j}'
        diagnostic_file = op.join(csv_dir, f'{job_id}_diagnostic.csv')
        assert op.exists(diagnostic_file), f'{diagnostic_file} : Not found !'

        # check variables 'divergent__', 'accept_stat__', 'lp__'
        print()
        print(f'Reading {op.basename(diagnostic_file)}... ', end='', flush=True)
        try:
            samples = stan.read_samples([diagnostic_file],
                                        variables_of_interest=['divergent__', 'accept_stat__', 'lp__'])
        except UnboundLocalError: # csv file contains no data
            print(f'NOK => looks like there is no data (check log file)')
            chains_status[j] = False
            continue
        print()

        # 1) check divergence (this is the most important)
        # there is one divergence for each sample (500 samples)
        # should be zero
        # if 1, means sampler did not sample from true posterior
        # TODO: if we find samples with 1, these samples have to be removed to build the posterior
        # if more than 5% of the samples have a divergence of 1, need to rerun the chain
        print(f'\tdivergence: ', end='', flush=True)
        divergent__ = samples['divergent__']
        divergent__sum = np.sum(divergent__)
        divergent__perc = 100 / 500 * divergent__sum
        print(f'{divergent__sum}/500 ({divergent__perc:.2f} %) of divergent samples => ', end='', flush=True)
        if divergent__perc > 5:
            print('NOK => rerun this chain !')
            chains_status[j] = False
        else:
            print('OK')

        # 2) check accept_stat__ (important but less than divergent__)
        # accept_stat__ is the percentage of accepted samples
        # if it is below 0.6 , need to rerun the chain
        accept_stat__ = samples['accept_stat__']
        print(f'\taccept_stat: mean across samples is {np.mean(accept_stat__):.2f} => ', end='', flush=True)
        if np.mean(accept_stat__) < 0.6:
            print('NOK => rerun this chain !')
            chains_status[j] = False
        else:
            print('OK')

        # 3) plot lp__
        # - what is bad is when lp stays constant (flat) for a long time, it means no new sample are generated (always taking the old one)
        # and the final number of different samples is very low (cf presentation)
        # - another bad thing is when lp is monotonically increasing ( because it should be stationary instead)
        # it means that number of warmup samples, supposed to reach a stationary distribution sampling, is not enough and should be increased
        lp__ = samples['lp__']

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        axes[0].plot(divergent__)
        axes[0].set_title('divergent__')
        axes[1].plot(accept_stat__)
        axes[1].set_title('accept_stat__')
        axes[1].set_ylim([0, 1])
        axes[2].plot(lp__)
        axes[2].set_title('lp__')
        fig.suptitle(f'chain {j}')

        fname = op.join(fig_dir, f'{job_id}_diagnostic.png')
        print(f'>> Saving {fname}...')
        fig.savefig(fname)

    print(f'Chain status : ')
    for c, cs in enumerate(chains_status):
        print(f'Chain {c} : {cs}')

    return chains_status

def chains_diagnostic_oldp(datafeatures_fname, stanmodel, n_init=8, n_chains_per_init=2):

    datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
    sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample', datafeatures_id)
    csv_dir = op.join(sample_output_dir, 'csv')
    fig_dir = op.join(csv_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    n_chains = n_init * n_chains_per_init
    chains_status = np.full((n_chains), True)

    for j in range(n_chains):
        job_id = f'{datafeatures_id}__j{j}'
        diagnostic_file = op.join(csv_dir, f'{job_id}_diagnostic.csv')
        assert op.exists(diagnostic_file), f'{diagnostic_file} : Not found !'

        # check variables 'divergent__', 'accept_stat__', 'lp__'
        print()
        print(f'Reading {op.basename(diagnostic_file)}... ', end='', flush=True)
        try:
            samples = stan.read_samples([diagnostic_file],
                                        variables_of_interest=['divergent__', 'accept_stat__', 'lp__'])
        except UnboundLocalError: # csv file contains no data
            print(f'NOK => looks like there is no data (check log file)')
            chains_status[j] = False
            continue
        print()

        # 1) check divergence (this is the most important)
        # there is one divergence for each sample (500 samples)
        # should be zero
        # if 1, means sampler did not sample from true posterior
        # TODO: if we find samples with 1, these samples have to be removed to build the posterior
        # if more than 5% of the samples have a divergence of 1, need to rerun the chain
        print(f'\tdivergence: ', end='', flush=True)
        divergent__ = samples['divergent__']
        divergent__sum = np.sum(divergent__)
        divergent__perc = 100 / 500 * divergent__sum
        print(f'{divergent__sum}/500 ({divergent__perc:.2f} %) of divergent samples => ', end='', flush=True)
        if divergent__perc > 5:
            print('NOK => rerun this chain !')
            chains_status[j] = False
        else:
            print('OK')

        # 2) check accept_stat__ (important but less than divergent__)
        # accept_stat__ is the percentage of accepted samples
        # if it is below 0.6 , need to rerun the chain
        accept_stat__ = samples['accept_stat__']
        print(f'\taccept_stat: mean across samples is {np.mean(accept_stat__):.2f} => ', end='', flush=True)
        if np.mean(accept_stat__) < 0.6:
            print('NOK => rerun this chain !')
            chains_status[j] = False
        else:
            print('OK')

        # 3) plot lp__
        # - what is bad is when lp stays constant (flat) for a long time, it means no new sample are generated (always taking the old one)
        # and the final number of different samples is very low (cf presentation)
        # - another bad thing is when lp is monotonically increasing ( because it should be stationary instead)
        # it means that number of warmup samples, supposed to reach a stationary distribution sampling, is not enough and should be increased
        lp__ = samples['lp__']

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        axes[0].plot(divergent__)
        axes[0].set_title('divergent__')
        axes[1].plot(accept_stat__)
        axes[1].set_title('accept_stat__')
        axes[1].set_ylim([0, 1])
        axes[2].plot(lp__)
        axes[2].set_title('lp__')
        fig.suptitle(f'chain {j}')

        fname = op.join(fig_dir, f'{job_id}_diagnostic.png')
        print(f'>> Saving {fname}...')
        fig.savefig(fname)

    print(f'Chain status : ')
    for c, cs in enumerate(chains_status):
        print(f'Chain {c} : {cs}')

    return chains_status




def extract_samples__variables_of_interest__parfor(datafeatures_fname, chains_indices, stanmodel,
                                                   variables_of_interest__csv=['log_lik'],
                                                   variables_of_interest__npz=['log_lik', 'x0', 'xhat_q', 'x']):
    @parfor(chains_indices, (datafeatures_fname, variables_of_interest__csv, variables_of_interest__npz, ), bar=False)
    def fun(j, datafeatures_fname, variables_of_interest__csv, variables_of_interest__npz):
        variables_of_interest = set(variables_of_interest__csv + variables_of_interest__npz)
        datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
        sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample_'+stanmodel, datafeatures_id)
        if not op.exists(sample_output_dir):
              sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample', datafeatures_id) 
        csv_dir = op.join(sample_output_dir, 'csv')
        job_id = f'{datafeatures_id}__j{j}'
        csv_fname = op.join(csv_dir, f'{job_id}.csv')
        print(f'>> Loading {csv_fname}...')
        samples = stan.read_samples([csv_fname], variables_of_interest=variables_of_interest)
        number_of_sensors = samples['xhat_q'].shape[2]
        csv_fnames = []
        for k in variables_of_interest__csv:
            v = samples[k]
            # remove the samples corresponding to the first 10 time samples => 10*(number of sensors)
            v = v[:, 10 * number_of_sensors:]
            # and save to csv
            v_fname = csv_fname.replace('.csv', f'__{k}.csv')
            print(f'>> Saving {v_fname}')
            v_df = pd.DataFrame(v)
            v_df.to_csv(v_fname, index=False)
            csv_fnames.append(v_fname)
        # save samples
        npz_fname = op.join(csv_dir, f'samples_{j}.npz')
        print(f'>> Saving {npz_fname}')
        np.savez(npz_fname, samples)
        return (csv_fnames, npz_fname, number_of_sensors)

    chains_number_of_sensors = [r[2] for r in fun]
    # assert number_of_sensors is the same across all chains
    assert np.all(np.array(chains_number_of_sensors) == chains_number_of_sensors[0])
    return fun


def write_r_code(csv_fnames):
    tmp = [op.dirname(f) for f in csv_fnames]
    assert np.all(np.array(tmp) == tmp[0])
    r_code_fname = op.join(tmp[0], 'r_hat_rstan.R')
    r_result_fname = op.join(tmp[0], 'r_hat_rstan.csv')
    fd = open(r_code_fname, 'w')
    fd.write('library(loo)' + '\n' + '\n')
    fd.write('\n'.join([f'print(paste("Reading", "{f}"))' + '\n' + f'll{i+1}=read.csv("{f}", header=T)' for i, f in enumerate(csv_fnames)]))
    fd.write('\n')
    fd.write('\n')
    fd.write('\n'.join([f'loo{i+1}=loo(as.matrix(ll{i+1}))' for i, _ in enumerate(csv_fnames)]))
    fd.write('\n')
    fd.write('\n')
    fd.write('lpd_point <- cbind(' + '\n' + '  ')
    fd.write(', \n  '.join([f'loo{i+1}$pointwise[,"elpd_loo"]' for i, _ in enumerate(csv_fnames)]))
    fd.write('\n' + '  )' + '\n')
    fd.write('\n')
    fd.write('pbma_wts <- pseudobma_weights(lpd_point, BB=FALSE)' + '\n')
    fd.write('pbma_BB_wts <- pseudobma_weights(lpd_point) # default is BB=TRUE' + '\n')
    fd.write('stacking_wts <- stacking_weights(lpd_point)' + '\n')
    fd.write('result=round(cbind( pbma_wts, pbma_BB_wts, stacking_wts), 2)' + '\n')
    fd.write('\n')
    fd.write(f'print(paste("Writing", "{r_result_fname}"))' + '\n')
    fd.write(f'write.csv(result, file="{r_result_fname}")')
    fd.write('\n')
    fd.write('\n')
    fd.close()
    return r_code_fname, r_result_fname


# def extract_samples__variables_of_interest(datafeatures_fname, chains):
#     datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
#     sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample', datafeatures_id)
#     csv_dir = op.join(sample_output_dir, 'csv')
#     for i, j in enumerate(chains['indices']):
#         job_id = f'{datafeatures_id}__j{j}'
#         csv_fname = op.join(csv_dir, f'{job_id}.csv')
#         print(f"Reading ({i+1}/{len(chains['indices'])}) {csv_fname}...")
#         samples = stan.read_samples([csv_fname], variables_of_interest=['x0', 'xhat_q', 'x']) # samples = stan.read_samples([csv_fname])
#         # save samples
#         fname = op.join(csv_dir, f'samples_{j}.npz')
#         print(f'>> Save {fname}')
#         np.savez(fname, samples)
#
#
# def extract_samples__variables_of_interest_parfor(datafeatures_fname, chains):
#     @parfor(chains['indices'], (datafeatures_fname,), bar=False):
#         sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample', datafeatures_id)
#         csv_dir = op.join(sample_output_dir, 'csv')
#         job_id = f'{datafeatures_id}__j{j}'
#         csv_fname = op.join(csv_dir, f'{job_id}.csv')
#         print(f"Loading {csv_fname}...")
#         samples = stan.read_samples([csv_fname], variables_of_interest=['x0', 'xhat_q', 'x']) # samples = stan.read_samples([csv_fname])
#         # save samples
#         fname = op.join(csv_dir, f'samples_{j}.npz')
#         print(f'>> Saving {fname}')
#         np.savez(fname, samples)
#         return fname
#     samples_fnames = [fname for fname in fun]
#     return samples_fnames


def compute_posterior(datafeatures_fname, chains, stanmodel):

    # ensure probability consistency
    # if len(chains['probabilities']) + 1 == len(chains['indices']):
    #     chains['probabilities'].append(1-np.sum(chains['probabilities']))
    assert len(chains['probabilities']) == len(chains['indices']) or \
           len(chains['probabilities']) + 1 == len(chains['indices']), f"len(chains['probabilities']): {len(chains['probabilities'])}," \
                                                                       f"len(chains['indices']): {len(chains['indices'])}"
    if len(chains['probabilities']) + 1 == len(chains['indices']):
        last_chain_probability = 1.0 - np.sum(chains['probabilities'])
        chains['probabilities'] = np.append(np.array(chains['probabilities']), last_chain_probability)
    # assert np.sum(chains['probabilities']) == 1.
    if not np.sum(chains['probabilities']) == 1.:
        print(f"!!! WARNING: np.sum(chains['probabilities']) = {np.sum(chains['probabilities'])}")

    datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
    sample_output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'sample_'+stanmodel, datafeatures_id)
    csv_dir = op.join(sample_output_dir, 'csv')
    chains_samples = []
    for i, j in enumerate(chains['indices']):
        # job_id = f'{datafeatures_id}__j{j}'
        # csv_fname = op.join(csv_dir, f'{job_id}.csv')
        # print(f'Reading {csv_fname}...')
        # samples = stan.read_samples([csv_fname], variables_of_interest=['x0', 'xhat_q', 'x']) # samples = stan.read_samples([csv_fname])
        # chains_samples.append(samples)
        # load samples
        fname = op.join(csv_dir, f'samples_{j}.npz')
        print(f">> Loading ({i + 1}/{len(chains['indices'])}) {fname}...")
        samples = np.load(fname, allow_pickle=True)['arr_0'][()]
        chains_samples.append(samples)

    # sample the "global" posterior distribution, by sampling distribution of each chain,
    # wrt their (previously) computed (using R loo) probability
    # x0sam_ = np.zeros((10000, 162))
    # x0sam = np.zeros((10000, 162))
    parameters = ['x0']
    posteriors = {}
    for p in parameters:
        # first dimension is #samples (500)
        assert chains_samples[0][p].shape[0] == 500
        posteriors[p] = np.zeros(tuple([10000] + [n for n in chains_samples[0][p].shape[1:]]))

    # for this, we need (to compute) the probability of each chain
    # this is done in R with r_hat_rstan_JD.R, using the (log-likelihood) files "ll0.csv", "ll2.csv", "ll4.csv", "ll6.csv"
    #       => probabilities are displayed when R script is over, using the stacking_weights method
    # p0 = 0.94
    # p2 = 0.0
    # p4 = 0.0
    # p6 = 1 - (p0 + p2 + p4)
    # p0 = chains['probabilities'][0]
    # p2 = chains['probabilities'][1]
    # p4 = chains['probabilities'][2]

    # x00 = chains_samples[0]['x0']
    # x02 = chains_samples[1]['x0']
    # x04 = chains_samples[2]['x0']
    # x06 = chains_samples[3]['x0']
    # xhatf0 = chains_samples[0]['xhat_q']
    # xhatf2 = chains_samples[1]['xhat_q']
    # xhatf4 = chains_samples[2]['xhat_q']
    # xhatf6 = chains_samples[3]['xhat_q']
    # xs0 = chains_samples[0]['x']
    # xs2 = chains_samples[1]['x']
    # xs4 = chains_samples[2]['x']
    # xs6 = chains_samples[3]['x']
    #
    # p0 = 0.94
    # p2 = 0.0
    # p4 = 0.0
    # p6 = 1 - (p0 + p2 + p4)

    cumulative_probabilities = [np.sum(chains['probabilities'][:c+1]) for c, _ in enumerate(chains['probabilities'])]

    # draw 10000 samples from the chains, using their respective probabilities
    # each sample is between 0 and 500 (500 is number of samples of each chain)
    j = np.random.randint(0, 500, 10000)
    for i in range(10000):
        # selection of the chain to use, based on their respective probabilities
        u = random.random()

        # if u < p0:
        #     chain_index_1 = 0
        #     x0sam_[i, :] = x00[j[i], :]
        # elif u < p0 + p2:
        #     chain_index_1 = 1
        #     x0sam_[i, :] = x02[j[i], :]
        # elif u < p0 + p2 + p4:
        #     chain_index_1 = 2
        #     x0sam_[i, :] = x04[j[i], :]
        # else:
        #     chain_index_1 = 3
        #     x0sam_[i, :] = x06[j[i], :]

        for c, p in enumerate(cumulative_probabilities):
            if u < p:
                chain_index_2 = c
                break
        # assert (chain_index_1 == chain_index_2) # or (chain_index_1 == 3 and chain_index_2 == -1)

        # x0sam[i, :] = chains_samples[chain_index_2]['x0'][j[i], :]
        for p in parameters:
            posteriors[p][i, :] = chains_samples[chain_index_2][p][j[i], :]

    # # assert np.all(x0sam_ == x0sam)
    # assert np.all(x0sam_ == posteriors['x0'])

    # save posterior
    for p in parameters:
        fname = op.join(csv_dir, f'posterior_{p}.npz')
        print(f'Save {fname}')
        np.savez(fname, posteriors[p])

    # plot posterior x0
    p = 'x0'
    plot_posterior_x0(posteriors[p], chains_dir=csv_dir)

    # plot data feature mean estimation at sensor and source level
    plot_data_feature_mean(chains, chains_samples, chains_dir=csv_dir)


def plot_posterior_x0(posterior, chains_dir):

    # p = 'x0'
    # fname = op.join(chains_dir, f'posterior_{p}.npz')
    # print(f'Load {fname}')
    # npzfile = np.load(fname)
    # x0sam = npzfile['arr_0']

    x0sam = posterior

    roi = vep_prepare_op.read_vep_mrtrix_lut()
    assert x0sam.shape[1] == 162
    x0sam_min, x0sam_max = np.min(x0sam)-0.2, np.max(x0sam)+0.2

    fig_dir = op.join(chains_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    figsize = (24, 12)
    for i in range(2):
        fig, ax = plt.subplots(figsize=figsize)

        regions_range = range(i*int(162/2), (i+1)*int(162/2))
        ax.violinplot(x0sam[:, regions_range], positions=range(len(regions_range)), showmedians=True)
        ax.set_ylim([x0sam_min, x0sam_max])
        ax.axhline(y=-3.0, color='r', linestyle='--', lw=0.5)
        ax.axhline(y=-2.05, color='r', linestyle='--', lw=0.5)
        ax.axhline(y=-3.9, color='r', linestyle='--', lw=0.5)

        ax.set_xticks(range(len(regions_range)))
        ax.set_xticklabels([roi[r] for r in regions_range], rotation='vertical')
        fig.suptitle(f'posterior distribution of excitabilities x0 ({i+1}/2)')
        fig.tight_layout()

        fname = op.join(fig_dir, f'posterior_x0_part{i+1}.png')
        print(f'Save {fname}')
        fig.savefig(fname)

    # compute the median
    x0_median = np.median(x0sam, axis=0)
    fname = op.join(chains_dir, 'posterior_x0_median.txt')
    print(f'Save {fname}')
    fd = open(fname, 'w')
    for r, l in enumerate(roi):
        txt = f'{l}, {x0_median[r]} (#{r + 1})'
        print(txt)
        fd.write(txt + '\n')
    fd.close()
    # sort the median
    x0_median_sorted_indices = np.argsort(x0_median)[::-1]
    fname = op.join(chains_dir, 'posterior_x0_median_sorted.txt')
    print(f'Save {fname}')
    fd = open(fname, 'w')
    for r in x0_median_sorted_indices:
        txt = f'{roi[r]}, {x0_median[r]} (#{r+1})'
        print(txt)
        fd.write(txt + '\n')
    fd.close()


    # plot the 20 highest median
    regions_range = x0_median_sorted_indices[:20]
    x0sam_regions_range_min = np.min(x0sam[:, regions_range]) - 0.2
    x0sam_regions_range_max = np.max(x0sam[:, regions_range]) + 0.2
    fig, ax = plt.subplots(figsize=figsize)
    ax.violinplot(x0sam[:, regions_range], positions=range(len(regions_range)), showmedians=True)
    ax.set_ylim([x0sam_regions_range_min, x0sam_regions_range_max])
    ax.axhline(y=-3.0, color='r', linestyle='--', lw=0.5)
    ax.axhline(y=-2.05, color='r', linestyle='--', lw=0.5)
    ax.axhline(y=-3.9, color='r', linestyle='--', lw=0.5)

    ax.set_xticks(range(len(regions_range)))
    ax.set_xticklabels([roi[r] for r in regions_range], rotation='vertical')
    fig.suptitle(f'posterior distribution of excitabilities x0 (20 regions with highest median x0)')
    fig.tight_layout()

    fname = op.join(fig_dir, f'posterior_x0_median_highest20.png')
    print(f'Save {fname}')
    fig.savefig(fname)


def plot_data_feature_mean(chains, chains_samples, chains_dir):

    chains_sensors_mean = np.array([chains['probabilities'][c] * chain_samples['xhat_q'].mean(axis=0)
                                    for c, chain_samples in enumerate(chains_samples)])
    xhatm = np.sum(chains_sensors_mean, axis=0)

    chains_sources_mean = np.array([chains['probabilities'][c] * chain_samples['x'].mean(axis=0)
                                    for c, chain_samples in enumerate(chains_samples)])
    xs = np.sum(chains_sources_mean, axis=0)

    # # REPLACE p1 p2 with poba of each chain
    # xhatm_check = p0 * xhatf0.mean(axis=0) + p2 * xhatf2.mean(axis=0) + \
    #               p4 * xhatf4.mean(axis=0) + p6 * xhatf6.mean(axis=0)
    # # sources
    # xm_check = p0 * xs0.mean(axis=0) + p2 * xs2.mean(axis=0) + \
    #            p4 * xs4.mean(axis=0) + p6 * xs6.mean(axis=0)

    # assert np.all(xhatm == xhatm_check)
    # assert np.all(xs == xm_check)


    # save results
    fname = op.join(chains_dir, 'data_feature_mean_sensor.npz')
    print(f'Save {fname}')
    np.savez(fname, xhatm)
    fname = op.join(chains_dir, 'data_feature_mean_source.npz')
    print(f'Save {fname}')
    np.savez(fname, xs)


    # and skip first 10 time samples
    fig_dir = op.join(chains_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    #assert xhatm.shape[0] == 501 and xs.T.shape[0] == 501
    times = range(10, xhatm.shape[0])
    fig, ax = plt.subplots()
    ax.plot(times, xhatm[times, :])
    ax.set_title('data_feature_mean_sensor')
    fname = op.join(fig_dir, 'data_feature_mean_sensor.png')
    print(f'Save {fname}')
    fig.savefig(fname)

    fig, ax = plt.subplots()
    ax.plot(times, xs.T[times, :])
    ax.set_title('data_feature_mean_source')
    fname = op.join(fig_dir, 'data_feature_mean_source.png')
    print(f'Save {fname}')
    fig.savefig(fname)
def findbest_sample_jobs(datafeatures_fname,
                                         n_optimize=50, n_best=8):

    # read csv files and get the 'lp__'
    datafeatures_id = op.basename(datafeatures_fname).rstrip('__datafeatures.R')
    output_dir = op.join(op.dirname(op.dirname(datafeatures_fname)), 'optimize', datafeatures_id)
    logs_dir = op.join(output_dir, 'logs')
    csv_dir = op.join(output_dir, 'csv')

    lp = np.full(n_optimize, np.nan)
    j_ok, j_nok, j_exc = 0, 0, 0

    for j in range(n_optimize):
        job_id = f'{datafeatures_id}__j{j}'
        log_fname = op.join(logs_dir, f'{job_id}.log')
        print(f'Reading status from {log_fname}... ', end='', flush=True)
        try:
            status = analyze_fit.analyze_log_file(log_fname)
        except Exception as e:
            print(e)
            j_exc += 1
            continue
        print(status)
        if status == 'ok':
            j_ok += 1
            csv_fname = op.join(csv_dir, f'{job_id}.csv')
            assert op.exists(csv_fname)
            print(f'\t\t>> Reading lp__ from {csv_fname}...', end=' ', flush=True)
            samples = stan.read_samples([csv_fname], variables_of_interest=['lp__'])
            print('ok')
            lp[j] = samples['lp__'][0]
        else:
            j_nok += 1

    assert j_ok + j_nok + j_exc == n_optimize
    print()
    print(f'>> Jobs: {n_optimize}, ok: {j_ok}, nok: {j_nok}, exc: {j_exc}')
    # HW: there is bug for trial patient, don't forget to change.
    sorted_indicies = np.argsort(-lp)
    n_best_indices = sorted_indicies[:n_best]
    return n_best_indices