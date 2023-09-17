In this tutorial, we start from a NumPy code and `torch.compile` it. To recap, the compilation here is a three-step process under the hood: first, NumPy constructs are converted into equivalent PyTorch calls; these Pytorch calls are then transpiled (lowered) into efficient C++ code, and finally these C++ kernels are JIT compiled into machine code. The result may give significant speedups as compared to original NumPy code, but
In this tutorial we consider a worked example of some of these transformations.

Before going into any details, a disclaimer: tricks and workarounds we describe did work on one specific example; while believe some of them can be useful in a wider context, they may or may not work in other contexts or require additional steps. 


# The setup

To be specific, we start from the code to construct the Mandelbrot fractal, which we take from N. Rougier "From python to NumPy" book, [https://www.labri.fr/perso/nrougier/from-python-to-numpy/], Chapter 4.3.

To recap, the Mandelbrot fractal is a locus of complex plane of the variable `c`, for which the iteration

$$
z_0 = 0, \qquad z_{n+1} = z_n^2 + c, \qquad n=0, 1, \dots
$$

remains bounded: $z_n < \infty$ as $n\to\infty$.

The code below constructs a grid of the size $x_n \times y_n$, defined by the box $[x_{min}, x_{max}] \times [y_{min}, y_{max}]$, and for each point of this mesh, $c$, performs `maxiter` steps of the  Mandelbrot iteration.


```python
import numpy as np

# from Chap 4.3, https://www.labri.fr/perso/nrougier/from-python-to-numpy/#temporal-vectorization

def mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):   
    X = np.linspace(xmin, xmax, xn, dtype='float32')
    Y = np.linspace(ymin, ymax, yn, dtype='float32')
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype='int')
    Z = np.zeros(C.shape, dtype='complex64')
    for n in range(maxiter):
        I = np.abs(Z) < horizon
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter-1] = 0
    return Z, N
```

Let's perform the iteration and visualize the result:


```python
xmin, xmax, xn = -2.25, +0.75, 3000 // 3 
ymin, ymax, yn = -1.25, +1.25, 2500 // 3
maxiter = 200
horizon = 2.0 ** 40

# iterate
Z, N = mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon)
```


```python
# plot a nice figure (see the original N. Rougier book for details)
def visualize(Z, N, horizon, xn, yn):
    log_horizon = math.log(horizon, 2)

    dpi = 72
    width = 10
    height = 10*yn/xn
                           
    # visualize (see the original version for details)
    M = np.nan_to_num(N + 1 - np.log2(np.log(abs(Z))) + log_horizon)

    dpi = 72
    width = 10
    height = 10*yn/xn

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)

    light = colors.LightSource(azdeg=315, altdeg=10)

    plt.imshow(light.shade(M, cmap=plt.cm.hot, vert_exag=1.5,
                           norm = colors.PowerNorm(0.3), blend_mode='hsv'),
               extent=[xmin, xmax, ymin, ymax], interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])
                                   
visualize(Z, N, horizon, xn, yn)
```

    /tmp/ipykernel_196042/2169256356.py:10: RuntimeWarning: invalid value encountered in log2
      M = np.nan_to_num(N + 1 - np.log2(np.log(abs(Z))) + log_horizon)



    
![png](steps_to_compile_mandelbrot_files/steps_to_compile_mandelbrot_4_1.png)
    


Let's now measure the runtime of this iteration (Our aim will be to improve the this using torch.compile).


```python
import time

def bench(func, name, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    print(f"{name} : ", end_time - start_time, "sec")

bench(mandelbrot_numpy, 'numpy', xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon)
```

    numpy :  1.5263571739196777 sec


# Torch.compile the iteration

To improve the run time of the algorithm, we will need to apply several identical transformations. Let us work it one by one.


## 1. Extract the inner iteration and only compile it

The first thing to note is that torch dynamo agressively unrolls loops. Thus compiling the `mandelbrot` function directly would unroll the inner loop of `maxiter` iterations. This leads to very long compile times and extravagant memory consumption; on consumer grade machines compilation may even run out of memory. To avoid this, we extract the inner iteration and will only compile it, not the whole simulation:


```python
def step(n, C, Z, N, horizon):
    I = np.abs(Z) < horizon
    N[I] = n
    Z[I] = Z[I]**2 + C[I]
    return Z, N


# note the additional argument, `step`
def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, step, horizon=2.0):   
    X = np.linspace(xmin, xmax, xn, dtype='float32')
    Y = np.linspace(ymin, ymax, yn, dtype='float32')
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype='int')
    Z = np.zeros(C.shape, dtype='complex64')
    for n in range(maxiter):
        Z, N = step(n, C, Z, N, horizon)       
    N[N == maxiter-1] = 0
    return Z, N
```


```python
# Set up the compilation

import torch

import torch._dynamo.config as cfg
cfg.numpy_ndarray_as_tensor = True    # this will go when torch_np is upstreamed

# make sure we all warnings from the compiler
import warnings
warnings.simplefilter('always')

# Uncomment these two lines below to see the compilation logs and more detailed feedback from the compiler
# (these will be relatively long and messy)
import logging
torch._logging.set_logs(dynamo=logging.WARNING)
```

The first attempt is to compile the `step` function as is. We will use the `fullgraph=True` argument to `torch.compile` so that the compilation fails on a graph break instead of falling back to eager mode.


```python
step_1 = torch.compile(fullgraph=True)(step)

# run it once to warm up the JIT
mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, step_1, horizon)
```


    ---------------------------------------------------------------------------

    DynamicOutputShapeException               Traceback (most recent call last)

    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:1353, in run_node(tracer, node, args, kwargs, nnmodule)
       1352 if op == "call_function":
    -> 1353     return node.target(*args, **kwargs)
       1354 elif op == "call_method":


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/utils/_stats.py:20, in count.<locals>.wrapper(*args, **kwargs)
         19 simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
    ---> 20 return fn(*args, **kwargs)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1160, in FakeTensorMode.__torch_dispatch__(self, func, types, args, kwargs)
       1159 try:
    -> 1160     return self.dispatch(func, types, args, kwargs)
       1161 except TypeError:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1380, in FakeTensorMode.dispatch(self, func, types, args, kwargs)
       1379 if run_impl_check(func):
    -> 1380     op_impl_out = op_impl(self, func, *args, **kwargs)
       1381     if op_impl_out != NotImplemented:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:554, in index_tensor(fake_mode, func, *args, **kwargs)
        553 with fake_mode:
    --> 554     out = meta_index_Tensor(*args, **kwargs)
        555     return out.to(out_device)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_meta_registrations.py:2477, in meta_index_Tensor(self, indices)
       2476 if index.dtype in [torch.int8, torch.bool]:
    -> 2477     nonzero = index.nonzero()
       2478     k = len(result)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/utils/_stats.py:20, in count.<locals>.wrapper(*args, **kwargs)
         19 simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
    ---> 20 return fn(*args, **kwargs)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1160, in FakeTensorMode.__torch_dispatch__(self, func, types, args, kwargs)
       1159 try:
    -> 1160     return self.dispatch(func, types, args, kwargs)
       1161 except TypeError:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1380, in FakeTensorMode.dispatch(self, func, types, args, kwargs)
       1379 if run_impl_check(func):
    -> 1380     op_impl_out = op_impl(self, func, *args, **kwargs)
       1381     if op_impl_out != NotImplemented:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:487, in nonzero(fake_mode, func, arg)
        482 if (
        483     fake_mode.shape_env is None
        484     or not fake_mode.shape_env.allow_dynamic_output_shape_ops
        485 ):
        486     # Without symints/symfloats, cannot handle this
    --> 487     raise DynamicOutputShapeException(func)
        489 if arg.nonzero_memo is None:


    DynamicOutputShapeException: aten.nonzero.default

    
    The above exception was the direct cause of the following exception:


    RuntimeError                              Traceback (most recent call last)

    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:1300, in get_fake_value(node, tx)
       1299     with tx.fake_mode, enable_python_dispatcher():
    -> 1300         return wrap_fake_exception(
       1301             lambda: run_node(tx.output, node, args, kwargs, nnmodule)
       1302         )
       1303 except Unsupported:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:890, in wrap_fake_exception(fn)
        889 try:
    --> 890     return fn()
        891 except UnsupportedFakeTensorException as e:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:1301, in get_fake_value.<locals>.<lambda>()
       1299     with tx.fake_mode, enable_python_dispatcher():
       1300         return wrap_fake_exception(
    -> 1301             lambda: run_node(tx.output, node, args, kwargs, nnmodule)
       1302         )
       1303 except Unsupported:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:1366, in run_node(tracer, node, args, kwargs, nnmodule)
       1365     fn_str = f"Failed running {op} {node.target}(*{args}, **{kwargs}):\n"
    -> 1366     raise RuntimeError(fn_str + str(e)).with_traceback(e.__traceback__) from e
       1368 raise AssertionError(op)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:1353, in run_node(tracer, node, args, kwargs, nnmodule)
       1352 if op == "call_function":
    -> 1353     return node.target(*args, **kwargs)
       1354 elif op == "call_method":


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/utils/_stats.py:20, in count.<locals>.wrapper(*args, **kwargs)
         19 simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
    ---> 20 return fn(*args, **kwargs)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1160, in FakeTensorMode.__torch_dispatch__(self, func, types, args, kwargs)
       1159 try:
    -> 1160     return self.dispatch(func, types, args, kwargs)
       1161 except TypeError:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1380, in FakeTensorMode.dispatch(self, func, types, args, kwargs)
       1379 if run_impl_check(func):
    -> 1380     op_impl_out = op_impl(self, func, *args, **kwargs)
       1381     if op_impl_out != NotImplemented:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:554, in index_tensor(fake_mode, func, *args, **kwargs)
        553 with fake_mode:
    --> 554     out = meta_index_Tensor(*args, **kwargs)
        555     return out.to(out_device)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_meta_registrations.py:2477, in meta_index_Tensor(self, indices)
       2476 if index.dtype in [torch.int8, torch.bool]:
    -> 2477     nonzero = index.nonzero()
       2478     k = len(result)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/utils/_stats.py:20, in count.<locals>.wrapper(*args, **kwargs)
         19 simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
    ---> 20 return fn(*args, **kwargs)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1160, in FakeTensorMode.__torch_dispatch__(self, func, types, args, kwargs)
       1159 try:
    -> 1160     return self.dispatch(func, types, args, kwargs)
       1161 except TypeError:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:1380, in FakeTensorMode.dispatch(self, func, types, args, kwargs)
       1379 if run_impl_check(func):
    -> 1380     op_impl_out = op_impl(self, func, *args, **kwargs)
       1381     if op_impl_out != NotImplemented:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_subclasses/fake_tensor.py:487, in nonzero(fake_mode, func, arg)
        482 if (
        483     fake_mode.shape_env is None
        484     or not fake_mode.shape_env.allow_dynamic_output_shape_ops
        485 ):
        486     # Without symints/symfloats, cannot handle this
    --> 487     raise DynamicOutputShapeException(func)
        489 if arg.nonzero_memo is None:


    RuntimeError: Failed running call_function <built-in function getitem>(*(FakeTensor(..., size=(833, 1000), dtype=torch.complex64), FakeTensor(..., size=(833, 1000), dtype=torch.bool)), **{}):
    aten.nonzero.default

    
    During handling of the above exception, another exception occurred:


    Unsupported                               Traceback (most recent call last)

    Cell In[7], line 4
          1 step_1 = torch.compile(fullgraph=True)(step)
          3 # run it once to warm up the JIT
    ----> 4 mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, step_1, horizon)


    Cell In[4], line 16, in mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, step, horizon)
         14 Z = np.zeros(C.shape, dtype='complex64')
         15 for n in range(maxiter):
    ---> 16     Z, N = step(n, C, Z, N, horizon)       
         17 N[N == maxiter-1] = 0
         18 return Z, N


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/eval_frame.py:294, in _TorchDynamoContext.__call__.<locals>._fn(*args, **kwargs)
        292 dynamic_ctx.__enter__()
        293 try:
    --> 294     return fn(*args, **kwargs)
        295 finally:
        296     set_eval_frame(prior)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/eval_frame.py:447, in catch_errors_wrapper.<locals>.catch_errors(frame, cache_size, frame_state)
        444             return hijacked_callback(frame, cache_size, hooks, frame_state)
        446 with compile_lock:
    --> 447     return callback(frame, cache_size, hooks, frame_state)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/convert_frame.py:128, in wrap_convert_context.<locals>._fn(*args, **kwargs)
        126 cleanup = setup_compile_debug()
        127 try:
    --> 128     return fn(*args, **kwargs)
        129 finally:
        130     cleanup.close()


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/convert_frame.py:364, in convert_frame_assert.<locals>._convert_frame_assert(frame, cache_size, hooks, frame_state)
        351 initial_torch_function_state = torch._C._is_torch_function_enabled()
        353 signpost_event(
        354     "dynamo",
        355     "_convert_frame_assert._compile",
       (...)
        361     },
        362 )
    --> 364 return _compile(
        365     frame.f_code,
        366     frame.f_globals,
        367     frame.f_locals,
        368     frame.f_builtins,
        369     compiler_fn,
        370     one_graph,
        371     export,
        372     export_constraints,
        373     hooks,
        374     frame,
        375     frame_state=frame_state,
        376 )


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:179, in dynamo_timed.<locals>.dynamo_timed_inner.<locals>.time_wrapper(*args, **kwargs)
        177 with torch.profiler.record_function(f"{key} (dynamo_timed)"):
        178     t0 = time.time()
    --> 179     r = func(*args, **kwargs)
        180     time_spent = time.time() - t0
        181 # print(f"Dynamo timer: key={key}, latency={latency:.2f} sec")


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/convert_frame.py:434, in _compile(code, globals, locals, builtins, compiler_fn, one_graph, export, export_constraints, hooks, frame, frame_state)
        432 for attempt in itertools.count():
        433     try:
    --> 434         out_code = transform_code_object(code, transform)
        435         orig_code_map[out_code] = code
        436         break


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/bytecode_transformation.py:1002, in transform_code_object(code, transformations, safe)
        999 instructions = cleaned_instructions(code, safe)
       1000 propagate_line_nums(instructions)
    -> 1002 transformations(instructions, code_options)
       1003 return clean_and_assemble_instructions(instructions, keys, code_options)[1]


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/convert_frame.py:419, in _compile.<locals>.transform(instructions, code_options)
        404 tracer = InstructionTranslator(
        405     instructions,
        406     code,
       (...)
        416     frame_state=frame_state,
        417 )
        418 with tracing(tracer.output.tracing_context):
    --> 419     tracer.run()
        420 output = tracer.output
        421 assert output is not None


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/symbolic_convert.py:2068, in InstructionTranslator.run(self)
       2067 def run(self):
    -> 2068     super().run()


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/symbolic_convert.py:727, in InstructionTranslatorBase.run(self)
        722 try:
        723     self.output.push_tx(self)
        724     while (
        725         self.instruction_pointer is not None
        726         and not self.output.should_exit
    --> 727         and self.step()
        728     ):
        729         pass
        730 except BackendCompilerFailed:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/symbolic_convert.py:687, in InstructionTranslatorBase.step(self)
        683         unimplemented(f"missing: {inst.opname}")
        684     TracingContext.set_current_loc(
        685         self.f_code.co_filename, self.lineno, self.f_code.co_name
        686     )
    --> 687     getattr(self, inst.opname)(inst)
        689     return inst.opname != "RETURN_VALUE"
        690 except BackendCompilerFailed:


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/symbolic_convert.py:395, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
        393 reason = None
        394 try:
    --> 395     return inner_fn(self, inst)
        396 except Unsupported as excp:
        397     if self.has_backedge() and self.should_compile_partial_graph():


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/symbolic_convert.py:166, in stack_op.<locals>.impl(self, inst)
        164 @functools.wraps(fn)
        165 def impl(self: "InstructionTranslatorBase", inst: Instruction):
    --> 166     self.push(fn_var.call_function(self, self.popn(nargs), {}))


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/variables/builtin.py:535, in BuiltinVariable.call_function(self, tx, args, kwargs)
        526     return wrap_fx_proxy_cls(
        527         UnspecializedPythonVariable,
        528         tx,
       (...)
        532         **options,
        533     )
        534 elif check_numpy_ndarray_args(args, kwargs):
    --> 535     return wrap_fx_proxy_cls(
        536         variables.NumpyNdarrayVariable,
        537         tx,
        538         proxy,
        539         **options,
        540     )
        541 elif all(isinstance(x, SymNodeVariable) for x in args):
        542     return SymNodeVariable.create(tx, proxy, None, **options)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/variables/builder.py:1178, in wrap_fx_proxy_cls(target_cls, tx, proxy, example_value, ignore_subclass, **options)
       1176 with preserve_rng_state():
       1177     if example_value is None:
    -> 1178         example_value = get_fake_value(proxy.node, tx)
       1180     # Handle recursive calls here
       1181     elif isinstance(example_value, FakeTensor):


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/utils.py:1317, in get_fake_value(node, tx)
       1313     unimplemented(f"data dependent operator: {cause.func}")
       1314 elif isinstance(
       1315     cause, torch._subclasses.fake_tensor.DynamicOutputShapeException
       1316 ):
    -> 1317     unimplemented(f"dynamic shape operator: {cause.func}")
       1318 elif isinstance(
       1319     cause, torch._subclasses.fake_tensor.UnsupportedOperatorException
       1320 ):
       1321     unimplemented(
       1322         f"unsupported operator: {cause.func} (see "
       1323         "https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0"
       1324         " for how to fix)"
       1325     )


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/exc.py:141, in unimplemented(msg)
        139 def unimplemented(msg: str):
        140     assert msg != os.environ.get("BREAK", False)
    --> 141     raise Unsupported(msg)


    Unsupported: dynamic shape operator: aten.nonzero.default
    
    from user code:
       File "/tmp/ipykernel_196042/4272750822.py", line 4, in step
        Z[I] = Z[I]**2 + C[I]
    
    Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
    
    
    You can suppress this exception and fall back to eager by setting:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True



Indeed, the compilations fails due to a graph break. The traceback is rather messy (unfortunately, this is endemic to JIT compiler tracebacks), but the gist is that it cannot properly handle the data dependent natire of the boolean indexing. Consider a very simple example:


```python
a = np.arange(3)
a[a % 2 == 0], a[a % 2 == 1]
```




    (array([0, 2]), array([1]))



Note that the size of the array indexed by a boolean array depends on the data values in the indexer array. This sort of behavior is too dynamic for the compiler to efficiently inline into C++ code (a more precise term here is _lowering_), so that the compiler gives up and falls back to the PyTorch eager mode. Had we not explicitly asked it to fail instead (by using `fullgraph=True` parameter), the result would be not any faster than the original NumPy code.

## 2. Remove data dependence from boolean indexing

Note that we only use boolean indices, `I`, as a mask into fixed-size arrays `Z` and `N`. We however, operate on arrays of a fixed size, and only assign elements of `Z` and `N` depending on masks. This allows us to identically rewrite our code to use `np.where` instead of the boolean indexing:


```python
def step_2(n, C, Z, N, horizon):
    I = np.abs(Z) < horizon
    N = np.where(I, n, N)            # N[I] = n
    Z = np.where(I, Z**2 + C, Z)     # Z[I] = Z[I]**2 + C[I]        
    return Z, N


def mandelbrot_2(xmin, xmax, ymin, ymax, xn, yn, maxiter, step, horizon=2.0):   
    X = np.linspace(xmin, xmax, xn, dtype='float32')
    Y = np.linspace(ymin, ymax, yn, dtype='float32')
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype='int')
    Z = np.zeros(C.shape, dtype='complex64')
    for n in range(maxiter):
        Z, N = step(n, C, Z, N, horizon)       
    N = np.where(N == maxiter-1, 0, N)        # N[N == maxiter-1] = 0
    return Z, N
```


```python
# compile and run the code (with a small number of iterations for now)
step_2 = torch.compile(step_2)
_ = mandelbrot_2(xmin, xmax, ymin, ymax, xn, yn, maxiter=10, step=step_2, horizon=horizon)
```

    /home/br/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_inductor/lowering.py:1302: UserWarning: Torchinductor does not support code generation for complex operators. Performance may be worse than eager.
      warnings.warn(
    Process ForkProcess-5:
    Process ForkProcess-8:
    Process ForkProcess-2:
    Process ForkProcess-7:
    Process ForkProcess-1:
    Traceback (most recent call last):
    Process ForkProcess-4:
    Process ForkProcess-3:
    Process ForkProcess-6:
    Traceback (most recent call last):
    Traceback (most recent call last):
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
    Traceback (most recent call last):
    Traceback (most recent call last):
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
    Traceback (most recent call last):
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
    Traceback (most recent call last):
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
    Traceback (most recent call last):
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
        self.run()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 96, in get
        with self._rlock:
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/synchronize.py", line 95, in __enter__
        return self._semlock.__enter__()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 96, in get
        with self._rlock:
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 96, in get
        with self._rlock:
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 97, in get
        res = self._recv_bytes()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/process.py", line 108, in run
        self._target(*self._args, **self._kwargs)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/connection.py", line 216, in recv_bytes
        buf = self._recv_bytes(maxlength)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/synchronize.py", line 95, in __enter__
        return self._semlock.__enter__()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/synchronize.py", line 95, in __enter__
        return self._semlock.__enter__()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/concurrent/futures/process.py", line 233, in _process_worker
        call_item = call_queue.get(block=True)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 96, in get
        with self._rlock:
    KeyboardInterrupt
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/synchronize.py", line 95, in __enter__
        return self._semlock.__enter__()
    KeyboardInterrupt
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 96, in get
        with self._rlock:
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 96, in get
        with self._rlock:
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/queues.py", line 96, in get
        with self._rlock:
    KeyboardInterrupt
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/synchronize.py", line 95, in __enter__
        return self._semlock.__enter__()
      File "/home/br/mambaforge/envs/torch_nightly/lib/python3.8/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)


## 3. Complex values not supported

The next obstacle is that our `C` array has the `complex64` dtype, and the compiler toolchain which transpiles our python code into efficient C++ does not handle complex numbers at the moment. The JIT compiler again falls back to the eager mode, and the performance is still not any better then the original.

To work around this limitation, we expand our arrays to add a length-2 dimension: instead of an complex-valued array of shape `(n1, n2)` we use an real-valued array of shape `(n1, n2, 2)`, where the last dimension holds real and imaginary parts separately.


```python
x = np.linspace(xmin, xmax, xn, dtype='float32')
y = np.linspace(ymin, ymax, yn, dtype='float32')

# instead of C = X[None, :] + 1j* Y[:, None]
c = np.stack(np.broadcast_arrays(x[None, :], y[:, None]), axis=-1)
c.shape
```




    (833, 1000, 2)



We then implement the absolute value and squaring manually:


```python
def abs2(a):
    r"""abs(a)**2 replacement."""
    return a[..., 0]**2 + a[..., 1]**2


def sq2(a):
    r"""a**2 replacement."""
    z = np.empty_like(a)
    z[..., 0] = a[..., 0]**2 - a[..., 1]**2
    z[..., 1] = 2 * a[..., 0] * a[..., 1]
    return z
```

Taking this all together, we have


```python
@torch.compile(fullgraph=True, dynamic=True)
def step_3(n, c, Z, N, horizon):
    I = abs2(Z) < horizon**2                      # Note: abs2
    N = np.where(I, n, N)                         
    Z = np.where(I[..., None], sq2(Z) + c, Z)     # Note: sq2
    return Z, N


def mandelbrot_3(xmin, xmax, ymin, ymax, xn, yn,  maxiter, step, horizon=2.0):
    x = np.linspace(xmin, xmax, xn, dtype='float32')
    y = np.linspace(ymin, ymax, yn, dtype='float32')

    c = np.stack(np.broadcast_arrays(x[None, :], y[:, None]), axis=-1)

    N = np.zeros(c.shape[:-1], dtype='int')
    Z = np.zeros_like(c, dtype='float32')

    for n in range(maxiter):
        Z, N = step(n, c, Z, N, horizon)
    N = np.where(N == maxiter-1, 0, N)
    return Z, N

```

Now try to run the simulation with the original value of `maxiter=200`:


```python
_ = mandelbrot_3(xmin, xmax, ymin, ymax, xn, yn, maxiter=maxiter, step=step_3, horizon=horizon)
```

    [2023-07-19 13:23:03,735] torch._dynamo.convert_frame: [WARNING] torch._dynamo hit config.cache_size_limit (64)
    [2023-07-19 13:23:03,735] torch._dynamo.convert_frame: [WARNING]    function: 'step_3' (/tmp/ipykernel_196042/4181140197.py:1)
    [2023-07-19 13:23:03,735] torch._dynamo.convert_frame: [WARNING] to diagnose recompilation issues, set env variable TORCHDYNAMO_REPORT_GUARD_FAILURES=1 and also see https://pytorch.org/docs/master/compile/troubleshooting.html.



    ---------------------------------------------------------------------------

    Unsupported                               Traceback (most recent call last)

    Cell In[20], line 1
    ----> 1 _ = mandelbrot_3(xmin, xmax, ymin, ymax, xn, yn, maxiter=maxiter, step=step_3, horizon=horizon)


    Cell In[16], line 19, in mandelbrot_3(xmin, xmax, ymin, ymax, xn, yn, maxiter, step, horizon)
         16 Z = np.zeros_like(c, dtype='float32')
         18 for n in range(maxiter):
    ---> 19     Z, N = step(n, c, Z, N, horizon)
         20 N = np.where(N == maxiter-1, 0, N)
         21 return Z, N


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/eval_frame.py:294, in _TorchDynamoContext.__call__.<locals>._fn(*args, **kwargs)
        292 dynamic_ctx.__enter__()
        293 try:
    --> 294     return fn(*args, **kwargs)
        295 finally:
        296     set_eval_frame(prior)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/eval_frame.py:447, in catch_errors_wrapper.<locals>.catch_errors(frame, cache_size, frame_state)
        444             return hijacked_callback(frame, cache_size, hooks, frame_state)
        446 with compile_lock:
    --> 447     return callback(frame, cache_size, hooks, frame_state)


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/convert_frame.py:128, in wrap_convert_context.<locals>._fn(*args, **kwargs)
        126 cleanup = setup_compile_debug()
        127 try:
    --> 128     return fn(*args, **kwargs)
        129 finally:
        130     cleanup.close()


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/convert_frame.py:337, in convert_frame_assert.<locals>._convert_frame_assert(frame, cache_size, hooks, frame_state)
        327     else:
        328         log.warning(
        329             "torch._dynamo hit config.cache_size_limit (%s)\n"
        330             "   function: %s\n"
       (...)
        335             troubleshooting_url,
        336         )
    --> 337     unimplemented("cache_size_limit reached")
        339 if not has_tensor_in_frame(frame):
        340     return None


    File ~/mambaforge/envs/torch_nightly/lib/python3.8/site-packages/torch/_dynamo/exc.py:141, in unimplemented(msg)
        139 def unimplemented(msg: str):
        140     assert msg != os.environ.get("BREAK", False)
    --> 141     raise Unsupported(msg)


    Unsupported: cache_size_limit reached


## 4. Chunk the iterations

We hit an internal torch Dynamo limitation: since it aggressively unrolls all loops, it recompiles and caches a separate version of the `step_3` routine for each value of integer `n` from zero to `maxiter`. 
There are several ways around it:
- we can increase the internal cache size, this may lead to extravagant memory consumption and very long compilation times; 
- we can remove the `fullgraph=True` compilation argument. Then once the cache size is reached, further iterations would fall back to the eager mode, and this has a detremental effect on performance;

A better way is to perform iterations in chunks, which we do below.


```python
@torch.compile(dynamic=True)
def step_4(n0, c, Z, N, horizon, chunksize):
    for j in range(chunksize):
        n = n0 + j                    # update the iteration counter
        I = abs2(Z) < horizon**2
        N = np.where(I, n, N)
        Z = np.where(I[..., None], sq2(Z) + c, Z)
    return Z, N


def mandelbrot_4(xmin, xmax, ymin, ymax, xn, yn,  maxiter, step, horizon=2.0):
    x = np.linspace(xmin, xmax, xn, dtype='float32')
    y = np.linspace(ymin, ymax, yn, dtype='float32')
    c = np.stack(np.broadcast_arrays(x[None, :], y[:, None]), axis=-1)

    N = np.zeros(c.shape[:-1], dtype='int')
    Z = np.zeros_like(c, dtype='float32')

    chunksize=10                                     # compile this many steps
    n_chunks = maxiter // chunksize

    for i_chunk in range(n_chunks):
        n0 = i_chunk*chunksize                       # update the iteration counter
        Z, N = step(n0, c, Z, N, horizon, chunksize)

    N = np.where(N == maxiter-1, 0, N)
    return Z, N
```

## Benchmark the final result

Finally, we are in position to benchmark our final result:


```python
# warm up the JIT
Z, N = mandelbrot_4(xmin, xmax, ymin, ymax, xn, yn, maxiter=maxiter, step=step_4, horizon=horizon)

bench(mandelbrot_4, 'compiled_4', xmin, xmax, ymin, ymax, xn, yn, maxiter, step_4, horizon)
```

    compiled_4 :  0.32459521293640137 sec



```python
# The speedup against NumPy:

1.526357 / 0.324595
```




    4.702342919638318



# Bonus: run your NumPy code on CUDA

Since our approach involves automatically converting NumPy calls into equivalent PyTorch calls, and given that PyTorch tensors can live on either CPU and or GPU, we can, in fact, _make our NumPy program run on GPU unchanged_. All we need to do it to set the PyTorch default device to CUDA:

```
import torch
torch.set_default_device("cuda")
```

With that, the `torch.compile`-d code would
- convert NumPy arrays into PyTorch tensors on GPU
- compile the manipulations with arrays into CUDA calls
- convert the GPU tensors into CPU NumPy arrays on exit (or a graph break)

Note that the last point is inescapable: NumPy arrays are always on CPU and do not have a notion of 'device'. Therefore, data transfer to/from device happens automatically and there is no user control over it. This is of course a double-edged sword: on one hand, all you need to do to unleash the CUDA power is to add a single line above. On the other hand, device transfers can be costly and the performance characteristics of the resulting code need to be measured: anecdotally, we have seen both slowdowns and speed-up, both by an order of magnitude, depending on the implementation details (for instance, the chunk size for the chunked iteration). 

# Recap

To summarize, we started with a NumPy program which performs the Mandelbrot iteration, and used `torch.compile` to speed it up. En route, we worked around several peculiarities of the `torch.compile` toolchain, including the lack of complex number support, difficulties with compiling the data dependent control flows, and agressive unrolling of loops during compilation.

With rather mild rewrites of the original code, we got a performance increase of more than 4 times. Note that the specific performance numbers may rather strongly depend on the problem size and other details (for instance, the chunk size for splitting the iterations). Anecdotally, in other programs we saw speedups ranging from 3 to 50 depending on the problem size relative to the cache size of the target machine. Performance tuning remains an experimental activity and the outcomes very much depend on details.

Finally, we note that our mitigation tricks may be equally applicable to NumPy and PyTorch programs.
