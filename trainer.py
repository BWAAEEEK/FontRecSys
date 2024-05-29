import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModel
from model import Model
from dataset import FeatureDataset
from tqdm import tqdm
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, config, model: Model, device):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.token_model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(self.device)
        for param in self.token_model.parameters():
            param.requires_grad = False

        self.hit20 = Accuracy(task="multiclass", top_k=20, num_classes=267).to(self.device)
        self.hit10 = Accuracy(task="multiclass", top_k=10, num_classes=267).to(self.device)
        self.hit5 = Accuracy(task="multiclass", top_k=5, num_classes=267).to(self.device)

    def train(self, epoch, train_dataset: FeatureDataset, writer: SummaryWriter):
        self.model.train()
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        data_iter = tqdm(train_loader, desc=f"{epoch} - Train")

        avg_loss = 0.0
        avg_hit20 = 0.0
        avg_hit10 = 0.0
        avg_hit5 = 0.0

        for i, data in enumerate(data_iter):
            result = data['result'].to(self.device)
            label = data["label"].to(self.device)
            image_data = {k: v.squeeze(1).to(self.device) for k, v in data["image"].items()}
            ppt_node = data["ppt_node"].to(self.device)
            text_data = {k: v.to(self.device) for k, v in data["text"].items()}

            text = self.token_model(input_ids=text_data['input_ids'], attention_mask=text_data['attention_mask'])
            feature = torch.cat([result, text.pooler_output], dim=1)

            train_result = self.model(feature, image_data, ppt_node)
            loss = self.criterion(train_result, label)

            self.optimizer.zero_grad()

            hit20 = self.hit20(train_result, label)
            hit10 = self.hit10(train_result, label)
            hit5 = self.hit5(train_result, label)

            loss.backward()
            self.optimizer.step()

            writer.add_scalar('train_loss/batch', loss, epoch * len(data_iter) + i)
            writer.add_scalar('train_hit20/batch', hit20, epoch * len(data_iter) + i)
            writer.add_scalar('train_hit10/batch', hit10, epoch * len(data_iter) + i)
            writer.add_scalar('train_hit5/batch', hit5, epoch * len(data_iter) + i)

            avg_loss += loss.item()
            avg_hit20 += hit20.item()
            avg_hit10 += hit10.item()
            avg_hit5 += hit5.item()

            post_fix = {"loss": loss.item(),
                        "avg_loss": avg_loss / (i + 1),
                        "hit20": hit20.item(),
                        "hit10": hit10.item(),
                        "hit5": hit5.item(),
                        "avg_hit20": avg_hit20 / (i + 1),
                        "avg_hit10": avg_hit10 / (i + 1),
                        "avg_hit5": avg_hit5 / (i + 1)}

            data_iter.set_postfix(post_fix)

        avg_loss /= len(data_iter)
        avg_hit20 /= len(data_iter)
        avg_hit10 /= len(data_iter)
        avg_hit5 /= len(data_iter)

        return avg_loss, avg_hit20, avg_hit10, avg_hit5

    def evaluation(self, epoch, eval_dataset: FeatureDataset, writer: SummaryWriter):
        self.model.eval()

        # evaluation에는 shuffle 할 필요 없어여
        eval_loader = DataLoader(eval_dataset, batch_size=self.config['batch_size'])

        data_iter = tqdm(eval_loader, desc=f"{epoch} - Eval")

        avg_loss = 0.0
        avg_hit20 = 0.0
        avg_hit10 = 0.0
        avg_hit5 = 0.0

        for i, data in enumerate(data_iter):
            with torch.no_grad():
                result = data['result'].to(self.device)
                label = data["label"].to(self.device)
                image_data = {k: v.to(self.device) for k, v in data["image"].items()}
                ppt_node = data["ppt_node"].to(self.device)
                text_data = {k: v.to(self.device) for k, v in data["text"].items()}

                text = self.token_model(input_ids=text_data['input_ids'], attention_mask=text_data['attention_mask'])
                feature = torch.cat([result, text.pooler_output], dim=1)

                eval_result = self.model(feature, image_data, ppt_node)

                loss = self.criterion(eval_result, label)

                # eval_result = torch.topk(eval_result, 10).indices.float()
                hit20 = self.hit20(eval_result, label)
                hit10 = self.hit10(eval_result, label)
                hit5 = self.hit5(eval_result, label)

                writer.add_scalar('eval_loss/batch', loss, epoch * len(data_iter) + i)
                writer.add_scalar('eval_hit20/batch', hit20, epoch * len(data_iter) + i)
                writer.add_scalar('eval_hit10/batch', hit10, epoch * len(data_iter) + i)
                writer.add_scalar('eval_hit5/batch', hit5, epoch * len(data_iter) + i)
                avg_loss += loss.item()

                avg_hit20 += hit20.item()
                avg_hit10 += hit10.item()
                avg_hit5 += hit5.item()

                post_fix = {"loss": loss.item(),
                            "avg_loss": avg_loss / (i + 1),
                            "hit20": hit20.item(),
                            "hit10": hit10.item(),
                            "hit5": hit5.item(),
                            "avg_hit20": avg_hit20 / (i + 1),
                            "avg_hit10": avg_hit10 / (i + 1),
                            "avg_hit5": avg_hit5 / (i + 1)}

                data_iter.set_postfix(post_fix)

        avg_loss /= len(data_iter)
        avg_hit20 /= len(data_iter)
        avg_hit10 /= len(data_iter)
        avg_hit5 /= len(data_iter)

        return avg_loss, avg_hit20, avg_hit10, avg_hit5

    def save(self, epoch, file_path='./output2/'):
        model_name = f"RecSys_model_state_dict_ep{epoch}.pth"
        # 모델 저장
        torch.save(self.model.state_dict(), file_path + model_name)
        print('Model saved to', file_path + model_name)