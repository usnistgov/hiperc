# Makefile for HiPerC code

all: cpu_diffusion cpu_spinodal gpu_diffusion gpu_spinodal gpu_manufactured
.PHONY: all

.PHONY: run
run: run_cpu_diffusion \
     run_cpu_spinodal  \
     run_gpu_diffusion \
     run_gpu_manufactured \
     run_gpu_spinodal


# CPU codes

cpu_diffusion_list := cpu-serial-diffusion \
                      cpu-openmp-diffusion \
                      cpu-tbb-diffusion

cpu_spinodal_list := cpu-openmp-spinodal

.PHONY: cpu_diffusion
cpu_diffusion:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir);)

.PHONY: run_cpu_diffusion
run_cpu_diffusion:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir) run;)

.PHONY: cpu_spinodal
cpu_spinodal:
	@$(foreach dir, $(cpu_spinodal_list), $(MAKE) -C $(dir);)

.PHONY: run_cpu_spinodal
run_cpu_spinodal:
	@$(foreach dir, $(cpu_spinodal_list), $(MAKE) -C $(dir) run;)


# GPU codes

gpu_diffusion_list := gpu-cuda-diffusion \
                      gpu-openacc-diffusion \
                      gpu-opencl-diffusion

gpu_spinodal_list := gpu-cuda-spinodal

gpu_manufactured_list := gpu-cuda-manufactured

.PHONY: gpu_diffusion
gpu_diffusion:
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir);)

.PHONY: run_gpu_diffusion
run_gpu_diffusion:
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir) run;)

.PHONY: gpu_spinodal
gpu_spinodal:
	@$(foreach dir, $(gpu_spinodal_list), $(MAKE) -C $(dir);)

.PHONY: run_gpu_spinodal
run_gpu_spinodal:
	@$(foreach dir, $(gpu_spinodal_list), $(MAKE) -C $(dir) run;)

.PHONY: gpu_manufactured
gpu_manufactured:
	@$(foreach dir, $(gpu_manufactured_list), $(MAKE) -C $(dir);)

.PHONY: run_gpu_manufactured
run_gpu_manufactured:
	@$(foreach dir, $(gpu_manufactured_list), $(MAKE) -C $(dir) run;)


# KNL codes

phi_diffusion_list := phi-openmp-diffusion

.PHONY: phi_diffusion
phi_diffusion:
	@$(foreach dir, $(phi_diffusion_list), $(MAKE) -C $(dir);)

.PHONY: run_phi_diffusion
run_phi_diffusion:
	@$(foreach dir, $(phi_diffusion_list), $(MAKE) -C $(dir) run;)


# Cleanup

.PHONY: clean
clean:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir) clean;)
	@$(foreach dir, $(cpu_spinodal_list),  $(MAKE) -C $(dir) clean;)
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir) clean;)
	@$(foreach dir, $(gpu_manufactured_list),  $(MAKE) -C $(dir) clean;)
	@$(foreach dir, $(gpu_spinodal_list),  $(MAKE) -C $(dir) clean;)
	$(MAKE) -C doc clean

.PHONY: cleanall
cleanall:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir) cleanall;)
	@$(foreach dir, $(cpu_spinodal_list),  $(MAKE) -C $(dir) cleanall;)
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir) cleanall;)
	@$(foreach dir, $(gpu_manufactured_list), $(MAKE) -C $(dir) cleanall;)
	@$(foreach dir, $(gpu_spinodal_list),  $(MAKE) -C $(dir) cleanall;)
