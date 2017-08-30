# Makefile for phasefield-accelerator-benchmark code

all: cpu_diffusion gpu_diffusion

.PHONY: all cpu_diffusion gpu_diffusion run_diffusion clean cleanall

run: run_cpu_diffusion run_gpu_diffusion

cpu_diffusion_list := cpu-analytic-diffusion \
                      cpu-serial-diffusion \
                      cpu-openmp-diffusion \
                      cpu-tbb-diffusion

gpu_diffusion_list := gpu-cuda-diffusion \
                      gpu-openacc-diffusion

cpu_diffusion:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir);)

gpu_diffusion:
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir);)

run_cpu_diffusion:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir) run;)

run_gpu_diffusion:
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir) run;)

clean:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir) clean;)
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir) clean;)

cleanall:
	@$(foreach dir, $(cpu_diffusion_list), $(MAKE) -C $(dir) cleanall;)
	@$(foreach dir, $(gpu_diffusion_list), $(MAKE) -C $(dir) cleanall;)
