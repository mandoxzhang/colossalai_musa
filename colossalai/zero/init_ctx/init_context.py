import functools

import torch
from colossalai.utils.cuda import get_current_device
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_param import ShardedParamV2
from colossalai.utils.memory_tracer.allocator import GLOBAL_MODEL_DATA_TRACER


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):

    def __init__(self):
        pass

    def __enter__(self):
        r"""
        Enter the context scope.
        """

        def preprocess_after(f):

            @functools.wraps(f)
            def wrapper(module: torch.nn.Module, *args, **kwargs):
                f(module, *args, **kwargs)
                self._post_init_method(module)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = preprocess_after(cls.__init__)

        # The function is called during init subclass.
        def _init_subclass(cls, **kwargs):
            cls.__init__ = preprocess_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        # Excution self._post_init_method after the default init function.
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _enable_class(subclass)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = (torch.nn.modules.module.Module.__init_subclass__)
        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

        self._pre_context_exec()

    def __exit__(self, exc_type, exc_value, traceback):

        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = (torch.nn.modules.module.Module._old_init_subclass)

        self._post_context_exec()
        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _pre_context_exec(self):
        pass

    def _post_context_exec(self):
        pass


class ZeroInitContext(InsertPostInitMethodToModuleSubClasses):
    r"""
    A context to initialize model.
    1. Convert the model to fp16.
    2. The paramaters of the module are adapted to type ShardedParameter.
    3. Shard the param and grad according to flags.
    rm_torch_payload_on_the_fly:
    True: remove tensor payload on param.data after module init finished.
    False: remove tensor payload on param.data afther the context exist.
            This is used when you add some logic to operate tensors in __init__ of module.
            See torchvision resnet18.
    """

    def __init__(self,
                 convert_fp16: bool,
                 convert_cuda: bool,
                 shard_strategy: BaseShardStrategy,
                 shard_param: bool = False,
                 shard_grad: bool = False,
                 rm_torch_payload_on_the_fly=False):
        super().__init__()
        self.convert_fp16 = convert_fp16
        self.convert_cuda = convert_cuda
        self.shard_param = shard_param
        self.shard_grad = shard_grad
        self.shard_strategy = shard_strategy
        self.rm_torch_payload_on_the_fly = rm_torch_payload_on_the_fly
        self.initialized_param_list = []

    def _post_context_exec(self):
        """The callback function when the context exits.
        """
        if not self.rm_torch_payload_on_the_fly:
            for param in self.initialized_param_list:
                assert hasattr(param, 'col_attr')
                param.col_attr.remove_torch_payload()

            del self.initialized_param_list

    def _post_init_method(self, module):
        r"""The function to call at the end of the constructor of each nn.Module.
        """
        for param in module.parameters():
            # avoid adapting a param to ShardedParam twice
            if hasattr(param, 'col_attr'):
                continue

            if self.convert_cuda:
                target_device = get_current_device()
            else:
                target_device = param.data.device

            # convert to fp16 and cuda if necessary
            if self.convert_fp16:
                param.data = param.data.to(torch.half).to(target_device)
                if param.grad is not None:
                    param.grad = param.grad.to(torch.half).to(target_device)

            param.col_attr = ShardedParamV2(param, rm_torch_payload=self.rm_torch_payload_on_the_fly)

            self.initialized_param_list.append(param)

            if self.shard_param:
                self.shard_strategy.shard(tensor_list=[param.col_attr._data_sharded_tensor])
                GLOBAL_MODEL_DATA_TRACER.trace_tensor(param.col_attr._data_sharded_tensor.payload)
            if param.col_attr.grad and self.shard_grad:
                self.shard_strategy.shard(tensor_list=[param.col_attr._grad_sharded_tensor])
                GLOBAL_MODEL_DATA_TRACER.trace_tensor(param.col_attr._grad_sharded_tensor.payload)