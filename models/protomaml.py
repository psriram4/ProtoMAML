import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import copy
import torch.nn.functional as F

class ProtoMAML(nn.Module):
    def __init__(self, num_train_steps=1, num_test_steps=1, embed_dim=64, lr=1e-3, lr_inner=0.01, \
        lr_output=0.01, adapt_lr=0.01, device=None):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.num_test_steps = num_test_steps
        self.embed_dim = embed_dim
        self.lr = lr 
        self.lr_inner = lr_inner
        self.lr_output = lr_output
        self.adapt_lr = adapt_lr
        self.device = device
        self.init_embedder()
        self.optimizers = self.configure_optimizers()

    def init_embedder(self):
        self.embedder = torchvision.models.DenseNet(
            growth_rate=32,
            block_config=(6, 6, 6, 6),
            bn_size=2,
            num_init_features=64,
            num_classes=self.embed_dim
        )
        self.embedder.to(self.device)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.embedder.parameters(), lr=self.lr)
        return optim

    def calculate_prototypes(self, embed_feats, labels):
        num_classes = labels.shape[1]
        prototypes = []

        gt_class = torch.argmax(labels, dim=1)
        for c in range(num_classes):
            prototype_c = embed_feats[torch.where(gt_class == c)].mean(dim=0)
            prototypes.append(prototype_c)
        
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes

    def compute_adapted_params(self, support_imgs, support_labels, mode="train"):
        embed_feats = self.embedder(support_imgs)
        prototypes = self.calculate_prototypes(embed_feats, support_labels)

        model_copy = copy.deepcopy(self.embedder)
        model_copy.train()
        model_copy.to(self.device)
        adapt_optim = torch.optim.SGD(model_copy.parameters(), lr=self.adapt_lr)
        adapt_optim.zero_grad()

        # print(prototypes.shape)

        init_linear_weight = 2 * prototypes
        init_linear_bias = -1 * torch.norm(prototypes, dim=1)**2

        linear_weight = init_linear_weight.detach().requires_grad_()
        linear_bias = init_linear_bias.detach().requires_grad_()

        linear_weight.to(self.device)
        linear_bias.to(self.device)

        num_steps = self.num_train_steps if mode == "train" else self.num_test_steps

        for _ in range(num_steps):
            feats = model_copy(support_imgs)
            outputs = F.linear(feats, linear_weight, linear_bias) # shape: (N*K, num_classes)
            loss = F.cross_entropy(outputs, support_labels)
            loss.backward()
            adapt_optim.step()

            linear_weight.data = linear_weight.data - linear_weight.grad * self.lr_output
            linear_bias.data = linear_bias.data - linear_bias.grad * self.lr_output

            linear_weight.grad.fill_(0)
            linear_bias.grad.fill_(0)
            adapt_optim.zero_grad()


        linear_weight = init_linear_weight + (linear_weight - init_linear_weight).detach()
        linear_bias = init_linear_bias + (linear_bias - init_linear_bias).detach()

        return model_copy, linear_weight, linear_bias


    def training_step(self, tasks, idxes):
        for batch in tasks:
            support_imgs, support_labels, query_imgs, query_labels = batch
            support_imgs = torch.permute(support_imgs.float(), (0, 3, 1, 2))
            finetuned_model = self.compute_adapted_params(support_imgs, support_labels, mode="train")
            finetuned_embedder, linear_weight, linear_bias = finetuned_model

            query_imgs = torch.permute(query_imgs.float(), (0, 3, 1, 2))
            feats = finetuned_embedder(query_imgs)
            outputs = F.linear(feats, linear_weight, linear_bias)

            loss = F.cross_entropy(outputs, query_labels)
            loss.backward()

            for real_param, copy_param in zip(self.embedder.parameters(), finetuned_embedder.parameters()):
                real_param.grad += copy_param.grad

        opt = self.optimizers
        opt.step()
        opt.zero_grad()

    def validation_step(self, tasks, idxs):
        accuracies = []
        losses = []
        for batch in tasks:
            support_imgs, support_labels, query_imgs, query_labels = batch
            support_imgs = torch.permute(support_imgs.float(), (0, 3, 1, 2))
            finetuned_model = self.compute_adapted_params(support_imgs, support_labels, mode="test")
            finetuned_embedder, linear_weight, linear_bias = finetuned_model

            query_imgs = torch.permute(query_imgs.float(), (0, 3, 1, 2))
            feats = finetuned_embedder(query_imgs)
            outputs = F.linear(feats, linear_weight, linear_bias)

            loss = F.cross_entropy(outputs, query_labels)
            acc = (outputs.argmax(dim=1) == query_labels.argmax(dim=1)).float()
        
            losses.append(loss.mean().detach())
            accuracies.append(acc.mean().detach())

        print(f"Validation Accuracy: {sum(accuracies)/len(accuracies)}")
        print(f"Validation Loss: {sum(losses)/len(losses)}")

        return sum(accuracies)/len(accuracies), sum(losses)/len(losses)
        
















# class ProtoMAML(pl.LightningModule):
#     def __init__(self, num_steps=1, embed_dim=64, lr=1e-3, adapt_lr=0.1):
#         super().__init__()
#         self.save_hyperparameters()
#         self.init_embedder()

#     def init_embedder(self):
#         self.embedder = torchvision.models.DenseNet(
#             growth_rate=32,
#             block_config=(6, 6, 6, 6),
#             bn_size=2,
#             num_init_features=64,
#             num_classes=self.hparams.embed_dim
#         )

#     def configure_optimizers(self):
#         optim = torch.optim.AdamW(self.embedder.parameters(), lr=self.hparams.lr)
#         return optim

#     def calculate_prototypes(self, embed_feats, labels):
#         prototypes = None
#         return prototypes

#     def compute_adapted_params(self, support_imgs, support_labels):
#         embed_feats = self.embedder(support_imgs)
#         prototypes = self.calculate_prototypes(embed_feats, support_labels)

#         model_copy = copy.deepcopy(self.embedder)
#         model_copy.train()
#         adapt_optim = torch.optim.SGD(model_copy.parameters(), lr=self.hparams.adapt_lr)
#         adapt_optim.zero_grad()

#         linear_weight = 2 * prototypes
#         linear_bias = -1 * torch.linalg.norm(prototypes, ord=2)

#         for i in range(self.hparams.num_steps):
#             feats = model_copy(support_imgs)
#             outputs = F.linear(feats, linear_weight, linear_bias)
#             loss = F.cross_entropy(outputs, support_labels)
#             loss.backward()
#             adapt_optim.step()

#             adapt_optim.zero_grad()

#         return model_copy, linear_weight, linear_bias


#     def training_step(self, batch, batch_idx):
#         support_imgs, support_labels, query_imgs, query_labels = batch

#         1/0
#         finetuned_model = self.compute_adapted_params(support_imgs, support_labels)
#         finetuned_embedder, linear_weight, linear_bias = finetuned_model

#         feats = finetuned_embedder(query_imgs)
#         outputs = F.linear(feats, linear_weight, linear_bias)
#         loss = F.cross_entropy(outputs, query_labels)
#         loss.backward()
#         adapt_optim.step()

#         loss.backward()
#         for p_global, p_local in zip(self.embedder.parameters(), finetuned_embedder.parameters()):
#             p_global.grad += p_local.grad

#         opt = self.optimizers()
#         opt.step()
#         opt.zero_grad()
        

#         pass

#     def forward(self, support, query):
#         # shapes for N-way K-shot classification (N classes, K examples per class):
#         # support set: (N*K, 84, 84, 3)






#         pass