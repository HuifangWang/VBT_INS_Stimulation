import os
import os.path as op
import vep_prepare_op
import numpy as np
import stan
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
try:
    matplotlib.use('Qt5Agg')
except:
    pass
plt.interactive(True)
import tempfile


roi = vep_prepare_op.read_vep_mrtrix_lut()

#
# vep_prepare.fitting_optimization_errors(pid, seizure_ind=None, opts=['vep_W'], save_fname=None)

def save_fig_bt(fig, output_dir, df_id, bt):
    fname = op.join(output_dir, 'figures', f'fit_analysis__{df_id}__bt_{bt}.png')
    print(f'>> Save {fname}')
    os.makedirs(op.dirname(fname), exist_ok=True)
    fig.savefig(fname)


def pid_files(pid, df_id, fit_id, option, bt_ind, parallel_mode, data_base_dir='/data'):
    pfd = vep_prepare_op.pid_fit_dir(pid, fit_id=fit_id, option=option, data_base_dir=data_base_dir)
    if parallel_mode:
        pfd_logs = op.join(pfd, 'parallel')
        # assert that there is one single directory...
        tmp_dir = os.listdir(pfd_logs)
        assert len(tmp_dir) == 1, f'One single directory expected, and got {tmp_dir}'
        pfd_logs = op.join(pfd_logs, tmp_dir[0], 'logs')
    else:
        pfd_logs = op.join(pfd, 'logs')

    if bt_ind is None:
        data_fname = op.join(pfd, f'Rfiles/fit_data_{df_id}.R')
        csv_fname = op.join(pfd, f'Cfiles/samples_{df_id}.csv')
        log_fname = op.join(pfd_logs, f'snsrfit_ode_{df_id}.log')
    else:
        data_fname = op.join(pfd, f'RfilesBT/fit_data_{df_id}_{bt_ind}.R')
        csv_fname = op.join(pfd, f'OptimalBT/{df_id}_{bt_ind}.csv')
        log_fname = op.join(pfd_logs, f'snsrfit_ode_{df_id}_{bt_ind}.log')
    return data_fname, csv_fname, log_fname


def read_iterations(csv_fname, last_iteration_csv_fname=None):

    print('>> Read', csv_fname, '... ', end='', flush=True)
    samples = pd.read_csv(csv_fname, dtype=np.dtype('float64'), skiprows=27)
    # print('OK')
    assert set(samples.dtypes) == {np.dtype('float64')}
    samples_np = samples.to_numpy(np.float64, copy=True)
    n_iter = samples_np.shape[0]
    print(f'OK ({n_iter} iterations detected)')

    lp_ind = samples.columns.get_loc('lp__')
    x0_ind = [c for c, col in enumerate(samples.columns) if 'x0.' in col]
    amplitude_ind = samples.columns.get_loc('amplitude')
    offset_ind = samples.columns.get_loc('offset')
    K_ind = samples.columns.get_loc('K')
    tau0_ind = samples.columns.get_loc('tau0')
    x_init_ind = [c for c, col in enumerate(samples.columns) if 'x_init.' in col]
    z_init_ind = [c for c, col in enumerate(samples.columns) if 'z_init.' in col]
    eps_slp_ind = samples.columns.get_loc('eps_slp')
    eps_snsr_pwr_ind = samples.columns.get_loc('eps_snsr_pwr')
    x_ind = [c for c, col in enumerate(samples.columns) if 'x.' in col]
    z_ind = [c for c, col in enumerate(samples.columns) if 'z.' in col]
    mu_slp_ind = [c for c, col in enumerate(samples.columns) if 'mu_slp.' in col]
    mu_snsr_pwr_ind = [c for c, col in enumerate(samples.columns) if 'mu_snsr_pwr.' in col]

    ns = len(x0_ind)
    assert ns == len(roi)
    # number of (bipolar) contacts
    nc = len(mu_snsr_pwr_ind)
    nc_nt = len(mu_slp_ind)
    assert np.mod(nc_nt, nc) == 0
    # number of time samples
    nt = round(nc_nt/nc)

    # check columns names
    column_names = ['lp__'] + \
                   [f'x0.{s}' for s in range(1, ns + 1)] + \
                   ['amplitude', 'offset', 'K', 'tau0'] + \
                   [f'x_init.{s}' for s in range(1, ns + 1)] + \
                   [f'z_init.{s}' for s in range(1, ns + 1)] + \
                   ['eps_slp', 'eps_snsr_pwr'] + \
                   [f'x.{t}.{s}' for s in range(1, ns + 1) for t in range(1, nt + 1)] + \
                   [f'z.{t}.{s}' for s in range(1, ns + 1) for t in range(1, nt + 1)] + \
                   [f'mu_slp.{t}.{c}' for c in range(1, nc + 1) for t in range(1, nt + 1)] + \
                   [f'mu_snsr_pwr.{c}' for c in range(1, nc + 1)]
    assert np.all(column_names == samples.columns)

    iterations = {}

    lp = samples_np[:, lp_ind]
    x0 = samples_np[:, x0_ind]
    amplitude = samples_np[:, amplitude_ind]
    offset = samples_np[:, offset_ind]
    K = samples_np[:, K_ind]
    tau0 = samples_np[:, tau0_ind]
    x_init = samples_np[:, x_init_ind]
    z_init = samples_np[:, z_init_ind]
    eps_slp = samples_np[:, eps_slp_ind]
    eps_snsr_pwr = samples_np[:, eps_snsr_pwr_ind]
    x = samples_np[:, x_ind].reshape(n_iter, ns, nt).swapaxes(1, 2)
    z = samples_np[:, x_ind].reshape(n_iter, ns, nt).swapaxes(1, 2)
    mu_slp = samples_np[:, mu_slp_ind].reshape(n_iter, nc, nt).swapaxes(1, 2)
    mu_snsr_pwr = samples_np[:, mu_snsr_pwr_ind]

    if last_iteration_csv_fname:
        samples_last_iteration = stan.read_samples([last_iteration_csv_fname])
        x0_last_iteration = samples_last_iteration['x0'].squeeze()
        assert np.all(abs(x0[-1] - x0_last_iteration).flatten() < 1e-15)
        mu_slp_last_iteration = samples_last_iteration['mu_slp'].squeeze()
        assert np.all(abs(mu_slp[-1] - mu_slp_last_iteration).flatten() < 1e-15)

    iterations['lp'] = lp
    iterations['x0'] = x0
    iterations['amplitude'] = amplitude
    iterations['offset'] = offset
    iterations['K'] = K
    iterations['tau0'] = tau0
    iterations['x_init'] = x_init
    iterations['z_init'] = z_init
    iterations['eps_slp'] = eps_slp
    iterations['eps_snsr_pwr'] = eps_snsr_pwr
    iterations['x'] = x
    iterations['z'] = z
    iterations['mu_slp'] = mu_slp
    iterations['mu_snsr_pwr'] = mu_snsr_pwr

    return iterations


def analysis_result(pid, pid_cr, df_id, fit_id, option, bt_ind, parallel_mode, data_base_dir='/data'):
    # get files
    data_fname, csv_fname, log_fname = pid_files(pid, df_id, fit_id, option, bt_ind, parallel_mode, data_base_dir=data_base_dir)
    # read data
    slp_data = stan.rload(data_fname)['slp']
    # read estimates
    mu_slp = read_iterations(csv_fname)['mu_slp'][-1, ...]
    # compute gof
    total_variance = np.sum(np.var(slp_data, axis=0))
    unexplained_variance = np.sum(np.var(slp_data - mu_slp, axis=0))
    gof_ = 100 * (1 - unexplained_variance / total_variance)
    # read convergence
    status = analyze_log_file(log_fname)
    return slp_data, mu_slp, gof_, status


log_status_ok = 'Optimization terminated normally:'
log_status_nok = 'Optimization terminated with error:'


def analyze_log_file(log_fname):
    fd = open(log_fname, 'r')
    lines = fd.readlines()
    fd.close()
    lines = [l.strip() for l in lines]
    if log_status_ok in lines:
        status = 'ok'
    elif log_status_nok in lines:
        status = 'nok'
    else:
        raise Exception(f'Unknown status for {log_fname}')
    return status


def analyze_fit_(pid, pid_cr, df_id, fit_id, option, output_dir, plot, plot_all_figures, close_fig, parallel_mode, data_base_dir='/data'):

    # columns = ['gof', 'status']
    results = np.full((1 + 100, 2), None)

    ## bt none
    print(f'>> Read bt_none')
    # get fit result
    slp_data, mu_slp, gof_, status = analysis_result(pid, pid_cr, df_id, fit_id, option, bt_ind=None, parallel_mode=parallel_mode, data_base_dir=data_base_dir)

    # plot result
    if plot:
        fig, axes = plt.subplots(nrows=2)
        axes[0].plot(slp_data)
        axes[0].set_title(f"{pid_cr} {df_id.replace(f'{pid}_ses-01_task-seizure_acq-', '')} - slp data")
        ylim = axes[0].get_ylim()
        # plot estimates
        axes[1].plot(mu_slp)
        axes[1].set_ylim(ylim)
        # plot convergence and gof as title
        axes[1].set_title(f'slp estimates, bt none, {round(gof_)}%', color='green' if status == 'ok' else 'red')
        fig.tight_layout()
        # save figure
        save_fig_bt(fig, output_dir, df_id, bt=None)
        if close_fig:
            plt.close(fig)
    # save result
    results[0] = (gof_, status)

    ## bt
    nrows, ncols = 5, 5
    fig, f = None, 0
    for bt_ind in np.arange(100):
        # print(f'>> Read bt_{bt_ind}')
        # get fit result
        slp_data, mu_slp, gof_, status = analysis_result(pid, pid_cr, df_id, fit_id, option, bt_ind=bt_ind, parallel_mode=parallel_mode, data_base_dir=data_base_dir)
        # plot result
        if plot and plot_all_figures:
            if fig is None or np.mod(bt_ind, nrows*ncols) == 0:
                if fig:
                    fig.tight_layout()
                    # save figure
                    save_fig_bt(fig, output_dir, df_id, bt=f)
                    if close_fig:
                        plt.close(fig)
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))
                f += 1
            ax = axes.flatten()[np.mod(bt_ind, nrows*ncols)]
            ax.plot(mu_slp)
            ax.set_ylim(ylim)
            ax.set_title(f'bt_{bt_ind}, {round(gof_)}%', color='green' if status == 'ok' else 'red')
        # save result
        results[1+bt_ind] = (gof_, status)

    if plot and plot_all_figures:
        if fig:
            fig.tight_layout()
            # save last figure
            save_fig_bt(fig, output_dir, df_id, bt=f)
            if close_fig:
                plt.close(fig)

    # save results
    index = pd.Index(['bt_none'] + [f'bt_{bt}' for bt in np.arange(100)], name='fit_analysis')
    df = pd.DataFrame(data=results, index=index, columns=pd.Index(['gof', 'status'], name='features'))
    fname = op.join(output_dir, 'csv', f'fit_analysis__{df_id}.csv')
    print(f'>> Save {fname}')
    os.makedirs(op.dirname(fname), exist_ok=True)
    df.to_csv(fname)

    return results


def analyze_fit(seizures, pid, pid_cr=None, fit_id=None, options=vep_prepare_op.options, report=True, plot=False, plot_all_figures=False, summarize=True, parallel_mode=True, data_base_dir='/data'):

    # if seizure_ind is not None:
    #     if isinstance(seizure_ind, int):
    #         seizure_ind = [seizure_ind]
    #     assert isinstance(seizure_ind, list), 'Argument seizure_ind must be specified as an index (or a list of indices)'

    # all_seizures = vep_prepare.get_all_seizures(pid, verbose=False)
    # seizures = [all_seizures[ind] for ind in seizure_ind] if seizure_ind else all_seizures
    # print('Seizures to be considered:')
    # for vhdr_fname in seizures:
    #     print(vhdr_fname)
    # print()

    summary = ''
    for _, vhdr_fname in enumerate(seizures):
        print(vhdr_fname)
        parameters = vep_prepare_op.load_data_feature_parameters(pid, fit_id, op.basename(vhdr_fname), data_base_dir=data_base_dir)
        hpf = parameters['hpf']
        lpf = parameters['lpf']
        # this is (very) dirty...
        df_id = op.basename(vhdr_fname).replace('_ieeg.vhdr', '')
        # in case this is a simulated seizure (.npz)
        df_id = df_id.rstrip('.npz')
        df_id = f"{df_id}_hpf{hpf}_lpf{lpf}"
        summary += f'{vhdr_fname}\n'
        for _, option in enumerate(options):
            print(option)
            output_dir = op.join(vep_prepare_op.pid_fit_dir(pid, fit_id=fit_id, option=option, data_base_dir=data_base_dir), 'fit_analysis')
            if report:
                analyze_fit_(pid, pid_cr, df_id, fit_id, option, output_dir, plot=plot, plot_all_figures=plot_all_figures, close_fig=True, parallel_mode=parallel_mode, data_base_dir=data_base_dir)
            if summarize:
                df_fname = op.join(output_dir, 'csv', f'fit_analysis__{df_id}.csv')
                summary_ = summarize_fit_(df_fname)
                print(summary_)
                summary += f'{option}\n{summary_}\n\n'
            print()
        summary += '\n'

    if summarize:
        print('\n\n>> Summary:')
        print(summary)
        pfd = vep_prepare_op.pid_fit_dir(pid, fit_id=fit_id, option='vep_dfs', data_base_dir=data_base_dir)
        fname = op.join(pfd, 'fit_analysis__summary.txt')
        print(f'>> Save {fname}')
        fd = open(fname, 'w')
        fd.write(summary)
        fd.close()


def summarize_fit_(df_fname):

    df = pd.read_csv(df_fname)
    bt_ok = np.nonzero(df.values[1:, -1] == 'ok')[0]
    bt_nok = np.nonzero(df.values[1:, -1] == 'nok')[0]
    summary = ''
    if bt_ok.size > 0:
        summary += f'bt_ok : {bt_ok.size}, mean(gof) = {np.mean(df.values[bt_ok, -2]):.2f} %'
    else:
        summary += f'bt_ok : none'
    summary += '\n'
    if bt_nok.size > 0:
        summary += f'bt_nok : {bt_nok.size}, mean(gof) = {np.mean(df.values[bt_nok, -2]):.2f} %'
    else:
        summary += f'bt_nok : none'
    return summary


# fit_ids is a list of dict
# for instance:
#       fit_ids = [{'fit_id': 'fit_V2', 'seizure_inds': [0, 1, 2, 3]},
#                  {'fit_id': 'fit_V3', 'seizure_inds': [2, 3]},]
#
def analyze_fits(pid, pid_cr, seizures, fit_ids, options, report=True, summarize=True, parallel_mode=True,
                 data_base_dir='/data'):
    for f in fit_ids:
        fit_id = f['fit_id']
        seizure_inds = f['seizure_inds']
        if report:
            for s in seizure_inds:
                for o in options:
                    print(f"analyze_fit of seizure {seizures[s]}, fit_id={fit_id}, option={o}")
                    analyze_fit([seizures[s]], pid, pid_cr=pid_cr, fit_id=fit_id, options=[o], report=True, plot=True, plot_all_figures=True, summarize=False, parallel_mode=parallel_mode, data_base_dir=data_base_dir)
        if summarize:
            analyze_fit([seizures[i] for i in seizure_inds], pid, pid_cr=pid_cr, fit_id=fit_id, options=options, report=False, plot=False, summarize=True, parallel_mode=parallel_mode, data_base_dir=data_base_dir)


def analyze_fits_report_parallel(pid, pid_cr, seizures, fit_ids, options, parallel_mode=True, 
                                 data_base_dir='/data', conda_path="/home/prior/anaconda3/etc/profile.d/conda.sh", nprocs=30):

    parallel_dir = op.join(vep_prepare_op.pid_fit_dir(pid, data_base_dir=data_base_dir), 'parallel', 'fit_analysis')
    os.makedirs(parallel_dir, exist_ok=True)
    parallel_dir = tempfile.mkdtemp(dir=parallel_dir)
    jobs_dir = op.join(parallel_dir, 'jobs')
    logs_dir = op.join(parallel_dir, 'logs')
    for d in [jobs_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    parallel_jobs_fname = op.join(jobs_dir, f'do_all_local.bash')
    print(f'>> Save list of jobs in {parallel_jobs_fname}')
    fd_parallel_jobs = open(parallel_jobs_fname, 'w')
    parallel_command = f'time /usr/bin/parallel -j {nprocs} < {parallel_jobs_fname}'
    fd_parallel_jobs.write(f'# {parallel_command}\n')

    report = True
    j = 0
    for f in fit_ids:
        fit_id = f['fit_id']
        seizure_inds = f['seizure_inds']
        if report:
            for s in seizure_inds:
                for o in options:
                    print(f"analyze_fit of seizure {seizures[s]}, fit_id={fit_id}, option={o}")
                    job_id = f"analyze_fit__{op.basename(seizures[s]).rstrip('_ieeg.vhdr')}__{fit_id}__{o}"
                    job_fname = op.join(jobs_dir, f'{job_id}.bash')
                    log_fname = op.join(logs_dir, f'{job_id}.log')
                    err_fname = op.join(logs_dir, f'{job_id}.err')
                    print(f'>> Save {job_fname}')
                    fd = open(job_fname, 'w')
                    fd.write('#!/bin/bash\n\n')
                    fd.write(f'. {conda_path}\n\n')
                    fd.write('conda activate vep\n\n')
                    fd.write(f'export PYTHONPATH={op.dirname(op.realpath(__file__))}\n\n')
                    # fd.write(f'python -c "print(\'coucou\')"')
                    fd.write(f"""python -c \"import analyze_fit; analyze_fit.analyze_fit(['{seizures[s]}'], '{pid}', """
                             f"""pid_cr='{pid_cr}', fit_id='{fit_id}', options=['{o}'], report=True, plot=True, plot_all_figures=True, summarize=False, parallel_mode={parallel_mode})\"                    """)
                    fd.close()
                    fd_parallel_jobs.write(f'bash {job_fname} > {log_fname} 2> {err_fname}\n')
                    j += 1

    fd_parallel_jobs.close()
    print()
    print(f'>> Save {j} jobs in {parallel_jobs_fname}')
    print(f'>> Run : {parallel_command}')
