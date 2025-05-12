
idbt := $(shell seq 0 $(N))
all_csv := $(patsubst %,$(RESULTS_DIR)/OptimalBT/$(FNAME_SUFFIX)_%.csv,$(idbt))

all: $(all_csv)

$(RESULTS_DIR)/OptimalBT/$(FNAME_SUFFIX)_%.csv: $(RESULTS_DIR)/RfilesBT/fit_data_$(FNAME_SUFFIX)_%.R $(RESULTS_DIR)/Rfiles/param_init_$(FNAME_SUFFIX).R
	./$(STAN_FNAME) optimize \
		algorithm=lbfgs tol_param=1e-4 iter=20000 save_iterations=0 \
		data file=$< \
		init=$(RESULTS_DIR)/Rfiles/param_init_$(FNAME_SUFFIX).R \
		output file=$@ refresh=10 \
		> $(RESULTS_DIR)/logs/snsrfit_ode_$(FNAME_SUFFIX)_$*.log 2>&1
