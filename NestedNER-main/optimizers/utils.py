from . import OPTIMIZER_REGISTRY
# 当我们 import optimizer 时，由于执行了这段代码，因此会先进行所有优化器的注册。之后，在调用 build_optimizer() 方法时就可以通过优化器名称从 OPTIMIZER_REGISTRY 中获取对应的优化器类进行实例化。
def build_optimizer(cfg, params):
    optimizer_name = cfg.optimizer
    optimizer = OPTIMIZER_REGISTRY[optimizer_name].build_optimizer(cfg, params)

    return optimizer
