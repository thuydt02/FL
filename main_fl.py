#AA00A3Y9Q3
import argparse

#from eFL import eFL
from federated_learning import Trainer

#NUM_CLIENTS = 300#300#1000 #300
BATCH_SIZE = 20
#LEARNING_RATE = 0.01 #0.01 for cifar10 , 0.01 for pre-training for metis graph
#LR_DECAY = 0.995

#GLOBAL_ROUNDS = 5000 # global for-loop interations 500 for FL

DATASET = "mnist"#"mnist"#"femnist" #"mnist"#"femnist"#"mnist"#"cifar100"#"femnist"

OUT_DIR = "/content/drive/MyDrive/eFL/output/"


w_DIR = OUT_DIR + DATASET + "/weight/"
z_DIR = OUT_DIR + DATASET + "/z_ass/"
acc_DIR = OUT_DIR + DATASET + "/acc/"


#PRE_TRAINED_W_FILE = "FL_non_iid_zipfz0.7_B10_L200_G20000.weight.pth"
#PRE_TRAINED_W_FILE = "FL_MLP2_non_iid_B10_L120_G500000.weight.pth"
PRE_TRAINED_W_FILE = None
#w_DIR ="../../weight/" # local
#z_DIR = "../../z_ass/" # local
#acc_DIR = "../../acc/" # local

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-clients', dest='num_clients', help='Number of clients', type=int, default=100)
    parser.add_argument('--lr', dest='lr', help='LEARNING RATE', type=float, default=0.003)
    parser.add_argument('--lr-decay', dest='lr_decay', help='LEARNING RATE DECAY', type=float, default=0.995)
    parser.add_argument('--client-epochs', dest='client_epochs', help='Number local epochs, default = 10', type=int, default=10)
    parser.add_argument('--server-epochs', dest='server_epochs', help='Number of epoch to train the FL model in server, default = 2000', type=int, default=2000)
    
    parser.add_argument('--partition-data-file', dest='partition_data_file', help='The partition data of clients', type=str, default="")
    
    return parser.parse_args()

def main(args):
    
    trainer = Trainer(n_clients = args.num_clients, learning_rate = args.lr, lr_decay = args.lr_decay, batch_size = BATCH_SIZE, 
        epochs = args.server_epochs, n_local_epochs = args.client_epochs, partition_data_file = args.partition_data_file,
        w_dir = w_DIR, acc_dir = acc_DIR, z_dir = z_DIR,
        dataset = DATASET, pre_trained_w_file = PRE_TRAINED_W_FILE)
    trainer.run_train()

if __name__ == '__main__':
    main(parse_args())



