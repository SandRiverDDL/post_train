/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/peft/tuners/lora/bnb.py:397: UserWarning: Merge lora module to 4-bit linear may get different generations due to rounding errors.
  warnings.warn(
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/chy/code/active/rl/scripts/train_grpo.py", line 189, in <module>
[rank0]:     main()
[rank0]:   File "/home/chy/code/active/rl/scripts/train_grpo.py", line 164, in main
[rank0]:     trainer.train()
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/trainer.py", line 2325, in train
[rank0]:     return inner_training_loop(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/trainer.py", line 2674, in _inner_training_loop
[rank0]:     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/trainer.py", line 4014, in training_step
[rank0]:     inputs = self._prepare_inputs(inputs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/extras/profiling.py", line 98, in wrapper
[rank0]:     return func(self, *args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1004, in _prepare_inputs
[rank0]:     generation_batch = self._generate_and_score_completions(generation_batch)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1374, in _generate_and_score_completions
[rank0]:     ) = self._generate(prompts, images)
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1316, in _generate
[rank0]:     prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, images)
[rank0]:                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1105, in _generate_single_turn
[rank0]:     self._move_model_to_vllm()
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/extras/profiling.py", line 98, in wrapper
[rank0]:     return func(self, *args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 953, in _move_model_to_vllm
[rank0]:     llm_model.load_weights([(name, param.data)])
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 332, in load_weights
[rank0]:     return loader.load_weights(weights)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/model_loader/online_quantization.py", line 173, in patched_model_load_weights
[rank0]:     return original_load_weights(auto_weight_loader, weights, mapper=mapper)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/models/utils.py", line 335, in load_weights
[rank0]:     autoloaded_weights = set(self._load_module("", self.module, weights))
[rank0]:                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/models/utils.py", line 288, in _load_module
[rank0]:     yield from self._load_module(
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/models/utils.py", line 261, in _load_module
[rank0]:     loaded_params = module_load_weights(weights)
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/models/qwen2.py", line 472, in load_weights
[rank0]:     weight_loader(param, loaded_weight, shard_id)
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/layers/linear.py", line 1044, in weight_loader_v2
[rank0]:     param.load_qkv_weight(
[rank0]:   File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/vllm/model_executor/parameter.py", line 200, in load_qkv_weight
[rank0]:     assert param_data.shape == loaded_weight.shape
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: AssertionError
  0%|                                                                                     | 0/1000 [00:02<?, ?it/s]
[rank0]:[W313 04:27:44.959308672 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())