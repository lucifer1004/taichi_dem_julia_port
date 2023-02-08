run:
	~/.local/bin/julia --project=. -t `nproc` script.jl

prepare-hpc:
	module load pyenv && \
	module load nvtools && \
	python3 -m pip install jill && \
	python3 -m jill install 1.9 --unstable --confirm && \
	python3 -m jill switch 1.9 && \
	~/.local/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'

profile:
	nvprof --profile-from-start off --openacc-profiling off julia --project=. script.jl