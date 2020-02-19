import pytorch_lightning as pl


class DefaultModel(pl.LightningModule):

    def forward(self, x):
        return self.model(x)

    @pl.data_loader
    def train_dataloader(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass
