from utils.train_utils import Trainer
import torch
import json

GLUE_QQP_DIR = "./data/QQP"
GLOVE_VECTORS_PATH = "./data/glove.6B.50d.txt"
with open('model_config.json') as json_file:
    CONFIG = json.load(json_file)

trainer = Trainer(
    glue_qqp_dir=GLUE_QQP_DIR,
    glove_vectors_path=GLOVE_VECTORS_PATH,
    min_token_occurancies=CONFIG["min_token_occurancies"],
    random_seed=CONFIG["random_seed"],
    emb_rand_uni_bound=CONFIG["emb_rand_uni_bound"],
    freeze_knrm_embeddings=CONFIG["freeze_knrm_embeddings"],
    knrm_kernel_num=CONFIG["knrm_kernel_num"],
    knrm_out_mlp=CONFIG["knrm_out_mlp"],
    dataloader_bs=CONFIG["dataloader_bs"],
    train_lr=CONFIG["train_lr"],
    change_train_loader_ep=CONFIG["change_train_loader_ep"]
)

epoch_states = trainer.train(CONFIG["n_epochs"])

# saving best model
# select epoch with max ndcg on, starting from 10th:
best_epoch = -1
best_ndcg = -1
start_epoch = 10 if CONFIG["n_epochs"] > 10 else 0
for i in range(start_epoch, CONFIG["n_epochs"]):
    if epoch_states[i][1] > best_ndcg:
        best_epoch = i
        best_ndcg = epoch_states[i][1]
torch.save(epoch_states[best_epoch][0], 'model_state/knrm.pth')

# save best model embeddings and vocab
torch.save(trainer.model.embeddings.state_dict(), 'model_state/emb_path_knrm.pth')
with open('model_state/vocab_path.json', 'w') as f:
    json.dump(trainer.vocab, f)