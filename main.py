from data.mini_imagenet_dataloader import MiniImageNetDataLoader
from models.protomaml import ProtoMAML
import pytorch_lightning as pl
from tqdm import tqdm
import torch

# HYPERPARAMETERS
total_train_step = 500
total_eval_step = 500
meta_batch_size = 4
num_train_steps = 5
num_test_steps = 10
embed_dim = 64
beta = 1e-3
alpha = 0.01
num_shot = 1
num_way = 5

device = None
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    
device = torch.device(device)
print(device)

dataloader = MiniImageNetDataLoader(shot_num=num_shot, way_num=num_way, episode_test_sample_num=15, device=device)

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='all')

model = ProtoMAML(num_train_steps=num_train_steps, num_test_steps=num_test_steps, embed_dim=embed_dim, lr=beta, \
    lr_inner=alpha, lr_output=alpha, adapt_lr=alpha, device=device)
loss = None

model.to(device)

# trainer = pl.Trainer()
# trainer.fit(model=model, train_dataloaders=dataloader)
val_idx = 0
task_idx = 0

for idx in tqdm(range(total_train_step)):
    # episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
    #     dataloader.get_batch(phase='train', idx=idx)

    tasks = []
    idxes = []
    
    # sample batch of tasks
    for _ in range(meta_batch_size):
        batch = dataloader.get_batch(phase='train', idx=task_idx)
        tasks.append(batch)
        idxes.append(task_idx)
        task_idx += 1
        
        
    # shapes for N-way K-shot classification (N classes, K examples per class):
    # episode_train_img: (N*K, 84, 84, 3)
    # episode_train_label: (N*K, N) --> one hot vector
    # episode_test_img: (Episode_test_samples*N, 84, 84, 3)
    # episode_test_label: (Episode_test_samples*N, N) --> one hot vector
    
    model.training_step(tasks, idxes)

    if idx % 20 == 0:
        tasks = []
        idxes = []
        for _ in range(meta_batch_size):
            batch = dataloader.get_batch(phase='val', idx=val_idx)
            tasks.append(batch)
            idxes.append(val_idx)
            val_idx += 1

        model.validation_step(tasks, val_idx)


    # if idx == 0:
    #     print("episode_train_img: ", episode_train_img.shape)
    #     print("episode_train_label: ", episode_train_label.shape)
    #     print("episode_test_img: ", episode_test_img.shape)
    #     print("episode_test_label: ", episode_test_label.shape)

task_idx = 0
eval_losses = []
eval_accs = []
for idx in tqdm(range(total_train_step)):
    tasks = []
    idxes = []
    
    # sample batch of tasks
    for _ in range(len(meta_batch_size)):
        batch = dataloader.get_batch(phase='test', idx=task_idx)
        tasks.append(batch)
        idxes.append(task_idx)
        task_idx += 1
        
        
    # shapes for N-way K-shot classification (N classes, K examples per class):
    # episode_train_img: (N*K, 84, 84, 3)
    # episode_train_label: (N*K, N) --> one hot vector
    # episode_test_img: (Episode_test_samples*N, 84, 84, 3)
    # episode_test_label: (Episode_test_samples*N, N) --> one hot vector
    
    acc, loss = model.validation_step(tasks, idxes)
    eval_losses.append(loss)
    eval_accs.append(acc)

print(f"Final Accuracy: {sum(eval_accs)/len(eval_accs)}")
print(f"Final Loss: {sum(eval_losses)/len(eval_losses)}")