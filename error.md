Traceback (most recent call last):
  File "/home/chy/code/active/rl/scripts/train_grpo.py", line 209, in <module>
    main()
  File "/home/chy/code/active/rl/scripts/train_grpo.py", line 184, in main
    trainer.train()
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/trainer.py", line 2325, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/trainer.py", line 2674, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/trainer.py", line 4014, in training_step
    inputs = self._prepare_inputs(inputs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/extras/profiling.py", line 98, in wrapper
    return func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1004, in _prepare_inputs
    generation_batch = self._generate_and_score_completions(generation_batch)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1374, in _generate_and_score_completions
    ) = self._generate(prompts, images)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1316, in _generate
    prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, images)
                                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 1292, in _generate_single_turn
    prompt_completion_ids = unwrapped_model.generate(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/peft/peft_model.py", line 2048, in generate
    outputs = self.base_model.generate(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 2566, in generate
    result = decoding_method(
             ^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 2786, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 494, in forward
    logits = self.lm_head(hidden_states[:, slice_indices, :])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chy/code/active/rl/.venv/lib/python3.11/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: expected scalar type Float but found BFloat16