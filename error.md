(EngineCore_DP0 pid=49825) [rank0]:W0314 05:16:54.001000 49825 torch/_inductor/utils.py:1613] [0/0] Not enough SMs to use max_autotune_gemm mode
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843] EngineCore failed to start.
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843] Traceback (most recent call last):
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 834, in run_engine_core
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     engine_core = EngineCoreProc(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 610, in __init__
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     super().__init__(
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 109, in __init__
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 235, in _initialize_kv_caches
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/executor/abstract.py", line 126, in determine_available_memory
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return self.collective_rpc("determine_available_memory")
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/executor/uniproc_executor.py", line 75, in collective_rpc
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/serial_utils.py", line 479, in run_method
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return func(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return func(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/worker/gpu_worker.py", line 324, in determine_available_memory
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     self.model_runner.profile_run()
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/worker/gpu_model_runner.py", line 4357, in profile_run
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     hidden_states, last_hidden_states = self._dummy_run(
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                                         ^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return func(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/worker/gpu_model_runner.py", line 4071, in _dummy_run
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     outputs = self.model(
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]               ^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/compilation/cuda_graph.py", line 126, in __call__
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return self.runnable(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 315, in forward
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     hidden_states = self.model(
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                     ^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/compilation/decorators.py", line 514, in __call__
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     output = TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/compilation/wrapper.py", line 171, in __call__
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return self._compiled_callable(*args, **kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 845, in compile_wrapper
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 990, in _compile_fx_inner
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     raise InductorError(e, currentframe()).with_traceback(
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 974, in _compile_fx_inner
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     mb_compiled_graph = fx_codegen_and_compile(
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                         ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 1695, in fx_codegen_and_compile
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 1505, in codegen_and_compile
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     compiled_module = graph.compile_to_module()
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                       ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/graph.py", line 2319, in compile_to_module
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return self._compile_to_module()
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/graph.py", line 2325, in _compile_to_module
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]                                                              ^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/graph.py", line 2271, in codegen
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     result = self.wrapper_code.generate(self.is_inference)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py", line 1552, in generate
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     return self._generate(is_inference)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py", line 1615, in _generate
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     self.generate_and_run_autotune_block()
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py", line 1695, in generate_and_run_autotune_block
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843]     raise RuntimeError(f"Failed to run autotuning code block: {e}") from e
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843] torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning code block: CUDA out of memory. Tried to allocate 594.00 MiB. GPU 0 has a total capacity of 11.94 GiB of which 7.38 GiB is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 3.28 GiB is allocated by PyTorch, and 69.69 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843] 
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843] Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
(EngineCore_DP0 pid=49825) ERROR 03-14 05:16:58 [core.py:843] 
(EngineCore_DP0 pid=49825) Process EngineCore_DP0:
(EngineCore_DP0 pid=49825) Traceback (most recent call last):
(EngineCore_DP0 pid=49825)   File "/home/chy/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore_DP0 pid=49825)     self.run()
(EngineCore_DP0 pid=49825)   File "/home/chy/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/process.py", line 108, in run
(EngineCore_DP0 pid=49825)     self._target(*self._args, **self._kwargs)
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 847, in run_engine_core
(EngineCore_DP0 pid=49825)     raise e
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 834, in run_engine_core
(EngineCore_DP0 pid=49825)     engine_core = EngineCoreProc(*args, **kwargs)
(EngineCore_DP0 pid=49825)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 610, in __init__
(EngineCore_DP0 pid=49825)     super().__init__(
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 109, in __init__
(EngineCore_DP0 pid=49825)     num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
(EngineCore_DP0 pid=49825)                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 235, in _initialize_kv_caches
(EngineCore_DP0 pid=49825)     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore_DP0 pid=49825)                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/executor/abstract.py", line 126, in determine_available_memory
(EngineCore_DP0 pid=49825)     return self.collective_rpc("determine_available_memory")
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/executor/uniproc_executor.py", line 75, in collective_rpc
(EngineCore_DP0 pid=49825)     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore_DP0 pid=49825)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/serial_utils.py", line 479, in run_method
(EngineCore_DP0 pid=49825)     return func(*args, **kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=49825)     return func(*args, **kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/worker/gpu_worker.py", line 324, in determine_available_memory
(EngineCore_DP0 pid=49825)     self.model_runner.profile_run()
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/worker/gpu_model_runner.py", line 4357, in profile_run
(EngineCore_DP0 pid=49825)     hidden_states, last_hidden_states = self._dummy_run(
(EngineCore_DP0 pid=49825)                                         ^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=49825)     return func(*args, **kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/worker/gpu_model_runner.py", line 4071, in _dummy_run
(EngineCore_DP0 pid=49825)     outputs = self.model(
(EngineCore_DP0 pid=49825)               ^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/compilation/cuda_graph.py", line 126, in __call__
(EngineCore_DP0 pid=49825)     return self.runnable(*args, **kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=49825)     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=49825)     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 315, in forward
(EngineCore_DP0 pid=49825)     hidden_states = self.model(
(EngineCore_DP0 pid=49825)                     ^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/compilation/decorators.py", line 514, in __call__
(EngineCore_DP0 pid=49825)     output = TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)
(EngineCore_DP0 pid=49825)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/compilation/wrapper.py", line 171, in __call__
(EngineCore_DP0 pid=49825)     return self._compiled_callable(*args, **kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 845, in compile_wrapper
(EngineCore_DP0 pid=49825)     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
(EngineCore_DP0 pid=49825)     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 990, in _compile_fx_inner
(EngineCore_DP0 pid=49825)     raise InductorError(e, currentframe()).with_traceback(
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 974, in _compile_fx_inner
(EngineCore_DP0 pid=49825)     mb_compiled_graph = fx_codegen_and_compile(
(EngineCore_DP0 pid=49825)                         ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 1695, in fx_codegen_and_compile
(EngineCore_DP0 pid=49825)     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 1505, in codegen_and_compile
(EngineCore_DP0 pid=49825)     compiled_module = graph.compile_to_module()
(EngineCore_DP0 pid=49825)                       ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/graph.py", line 2319, in compile_to_module
(EngineCore_DP0 pid=49825)     return self._compile_to_module()
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/graph.py", line 2325, in _compile_to_module
(EngineCore_DP0 pid=49825)     self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
(EngineCore_DP0 pid=49825)                                                              ^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/graph.py", line 2271, in codegen
(EngineCore_DP0 pid=49825)     result = self.wrapper_code.generate(self.is_inference)
(EngineCore_DP0 pid=49825)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py", line 1552, in generate
(EngineCore_DP0 pid=49825)     return self._generate(is_inference)
(EngineCore_DP0 pid=49825)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py", line 1615, in _generate
(EngineCore_DP0 pid=49825)     self.generate_and_run_autotune_block()
(EngineCore_DP0 pid=49825)   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py", line 1695, in generate_and_run_autotune_block
(EngineCore_DP0 pid=49825)     raise RuntimeError(f"Failed to run autotuning code block: {e}") from e
(EngineCore_DP0 pid=49825) torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning code block: CUDA out of memory. Tried to allocate 594.00 MiB. GPU 0 has a total capacity of 11.94 GiB of which 7.38 GiB is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 3.28 GiB is allocated by PyTorch, and 69.69 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
(EngineCore_DP0 pid=49825) 
(EngineCore_DP0 pid=49825) Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
(EngineCore_DP0 pid=49825) 
Traceback (most recent call last):
  File "/home/chy/code/active/rl/scripts/eval_dataset.py", line 137, in <module>
    main()
  File "/home/chy/code/active/rl/scripts/eval_dataset.py", line 93, in main
    result = run_harness_eval(
             ^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/src/rl/harness_tasks.py", line 166, in run_harness_eval
    return simple_evaluate(
           ^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/lm_eval/utils.py", line 498, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/lm_eval/evaluator.py", line 239, in simple_evaluate
    lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/lm_eval/api/model.py", line 180, in create_from_arg_obj
    return cls(**arg_dict, **additional_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/lm_eval/models/vllm_causallms.py", line 202, in __init__
    self.model = LLM(**self.model_args)  # type: ignore[invalid-argument-type]
                 ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/entrypoints/llm.py", line 334, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 183, in from_engine_args
    return cls(
           ^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 109, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 93, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 642, in __init__
    super().__init__(
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 471, in __init__
    with launch_core_engines(vllm_config, executor_class, log_stats) as (
  File "/home/chy/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/utils.py", line 903, in launch_core_engines
    wait_for_engine_startup(
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/v1/engine/utils.py", line 960, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}