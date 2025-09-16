import cmdstanpy

def ensure_cmdstan():
    try:
        cmdstanpy.cmdstan_path()
        print("Cmdstan already available, skipping installation.")
    except ValueError:
        print("Installing cmdstan...")
        cmdstanpy.install_cmdstan()
        print("Finished installing cmdstan.")

ensure_cmdstan()