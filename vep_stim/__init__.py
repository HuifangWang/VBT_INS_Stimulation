import cmdstanpy

def ensure_cmdstan():
    try:
        cmdstanpy.cmdstan_path()
        print("Cmdstan available on this system.")
    except ValueError as e:
        print("Cmdstan is not available on this system. Please install Cmdstan before proceeding.")

ensure_cmdstan()