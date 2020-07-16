# class LPRegularize(Callback):
#
#
#     def on_batch_end(self, trainer: Trainer, pl_module: LightningModule):
#         trainer.callback_metrics
#
#         if self.l1_strength is not None:
#             l1_reg = torch.tensor(0.)
#             for param in pl_module.parameters():
#                 l1_reg += torch.norm(param, 1)
#             loss += self.l1_strength * l1_reg
#
#         # L2 regularizer
#         if self.l2_strength is not None:
#             l2_reg = torch.tensor(0.)
#             for param in pl_module.parameters():
#                 l2_reg += torch.norm(param, 2)
#             loss += self.l2_strength * l2_reg
