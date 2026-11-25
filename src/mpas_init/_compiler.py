"""Experimental assertion-preserving CPU compiler for the tensor path.

Default Inductor in Torch 2.8 removes assertions and rewrites scalar arithmetic.
Keep sensitive operations in native Torch, and bound numerical region size.
This backend does not qualify untested profiles, precisions, devices, or AD.
"""

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._guards import TracingContext, detect_fake_mode
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Interpreter
from torch.fx.passes.split_module import split_module

_MAX_NODES = 4096


def _compile_regions(graph, inputs):
    checks = {torch._assert_async, torch.ops.aten._assert_async.msg, torch.ops.aten._functional_assert_async.msg}
    constants = {}

    def constant(value):
        if not isinstance(value, torch.fx.Node):
            return True
        if value not in constants:
            constants[value] = (value.op != "placeholder"
                                and all(constant(x) for x in value.all_input_nodes))
        return constants[value]

    def native(node):
        if node.op != "call_function":
            return False
        # Scalar libm and ATen trigonometry differ before cancelling winds.
        if node.target in checks or node.target in (torch.ops.aten.sin.default, torch.ops.aten.cos.default):
            return True
        # Reciprocal multiplication and repeated powers alter cancelling winds.
        if node.target in (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar):
            return constant(node.args[1])
        return (node.target == torch.ops.aten.pow.Tensor_Scalar
                and node.args[1] not in (0, 1, 2))

    native_nodes = {node.name for node in graph.graph.nodes if native(node)}
    current, length = 0, 0

    def partition(node):
        nonlocal current, length
        if node.name in native_nodes:
            current += 1
            result = current
            current += 1
            length = 0
            return result
        if length >= _MAX_NODES:
            current += 1
            length = 0
        length += 1
        return current

    split = split_module(graph, graph, partition, keep_original_order=True, keep_original_node_name=True)

    class CompileRegions(Interpreter):
        def call_module(self, target, arguments, keywords):
            child = self.fetch_attr(target)
            result = super().call_module(target, arguments, keywords)
            if not any(n.op == "call_function" and n.name in native_nodes for n in child.graph.nodes):
                if keywords:
                    raise ValueError("compiled FX regions require positional arguments")
                output = list(child.graph.nodes)[-1]
                single = not isinstance(output.args[0], (tuple, list))
                if single:
                    output.args = ((output.args[0],),)
                    child.recompile()
                # Region reports must not overwrite the whole graph's strides.
                with config.patch({"cpp.simdlen": 1, "keep_output_stride": True}), TracingContext.report_output_strides():
                    compiled = compile_fx_inner(child, list(arguments))

                def execute(*values):
                    outputs = compiled(list(values))
                    return outputs[0] if single else outputs

                child.forward = execute
            return result

    # Lazy nested compile wrappers can execute without invoking their backend.
    # Compile directly while propagating fake values, before any real execution.
    mode = detect_fake_mode(inputs) or FakeTensorMode()
    fake_inputs = [mode.from_tensor(x) if isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor) else x
                   for x in inputs]
    with mode:
        CompileRegions(split).run(*fake_inputs)
    return make_boxed_func(split.forward)


def checked_backend(graph, inputs):
    """Normalize AD first, then compile arithmetic and retain native assertions."""
    checks = {torch.ops.aten._assert_async.msg, torch.ops.aten._functional_assert_async.msg}
    decompositions = {op: function for op, function in select_decomp_table().items() if op not in checks}
    # Partitioning an open JVP context causes unsupported nested forward AD.
    # AOT normalization balances that context before numerical regions split.
    return aot_autograd(fw_compiler=_compile_regions, bw_compiler=_compile_regions,
                        decompositions=decompositions)(graph, inputs)
