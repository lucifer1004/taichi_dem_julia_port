run:
	LOCAL=true ~/.local/bin/julia --project=. script.jl

inspect:
	cat output.p4c | pcregrep -M "CONTACTS\n.*"

hpc: prepare-hpc run-hpc

run-hpc:
	/opt/julias/julia-1.9/bin/julia --project=. script.jl

prepare-hpc:
	sed 's/archive.ubuntu.com/mirrors.pku.edu.cn/g' -i /etc/apt/sources.list && \
	apt update && apt install -y xvfb && \
	nohup Xvfb -s '-screen 0 1024x768x24' & \
	python3 -m pip install jill && \
	python3 -m jill install 1.9 --unstable --confirm && \
	/opt/julias/julia-1.9/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'

profile:
	nvprof --profile-from-start off --openacc-profiling off julia --project=. script.jl