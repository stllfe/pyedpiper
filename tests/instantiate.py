

def test():
    from utils import instantiate
    from pytorch_lightning import Trainer
    from omegaconf import DictConfig

    config = DictConfig({'class': 'Trainer', 'module': 'pytorch_lightning', 'params': {'max_epochs': 10}})
    trainer = instantiate(config)
    assert isinstance(trainer, Trainer)


if __name__ == "__main__":
    test()
