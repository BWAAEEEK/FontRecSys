import torch
import numpy as np
import random
import datetime
from torch.utils.data import random_split
from dataset import FeatureDataset, FinalDataset
from model import Model
from trainer import Trainer
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
import pickle
from utils import image_processing
from transformers import ViTImageProcessor


now = datetime.datetime.now()
s = now.strftime("%Y-%m-%d-%H-%M-%S")


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = EasyDict()

    args.input_size = 780
    args.hidden_size = 600
    args.latent_size = 300
    args.learning_rate = 0.0001
    args.batch_size = 10
    args.epoch = 100
    args.seed = 42

    fix_seed(args.seed)

    config = vars(args)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print("Loading Dataset ...")
    with open('../Data/data_collection/MangoCrawling/mango_preprocessing.pkl', 'rb') as fr:
        data = pickle.load(fr)

    processor = ViTImageProcessor("facebook/sam-vit-base")
    image_data = image_processing(processor)

    datasets_size = len(data)
    train_size = int(datasets_size*0.8)
    validation_size = int(datasets_size*0.1)
    test_size = datasets_size - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(data, [train_size, validation_size, test_size])

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")
    node_idx = {k: idx for idx, k in enumerate(image_data.keys())}

    print("Building DataLoader ...")
    train_dataset = FeatureDataset(train_dataset, image_data, node_idx)
    test_dataset = FeatureDataset(test_dataset, image_data, node_idx)
    validation_dataset = FeatureDataset(validation_dataset, image_data, node_idx)

    print("Building Model ...")
    model = Model(config)

    print("Building Trainer ...")
    trainer = Trainer(config, model, device)

    print("+--------------------------------------------+")
    print("|               Training Start               |")
    print("+--------------------------------------------+")

    writer = SummaryWriter("./runs/{}_MSE_lr{}_batch{}_hidden{}_lay3".format(s,
                                                                             config["learning_rate"],
                                                                             config["batch_size"],
                                                                             config["hidden_size"]))
    stopping_epoch = 10
    stopping_epoch_s = 0
    be_loss = 10000

    # epoch 6에서 최고 성능
    for epoch in range(config["epoch"]):
        train_loss = trainer.train(epoch, train_dataset, writer)
        val_loss = trainer.evaluation(epoch, validation_dataset, writer)

        if val_loss > be_loss:
            stopping_epoch_s += 1
            print('@', val_loss, be_loss)

            be_loss = val_loss

        if stopping_epoch_s == stopping_epoch :
            break

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        trainer.save(epoch=epoch)

        test_acc = trainer.evaluation(epoch, test_dataset)
        print('\n\n')

