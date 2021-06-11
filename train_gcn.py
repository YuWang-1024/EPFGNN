import torch
from utils import load_experiment_configs
from trainers.trainer import Trainer
from models.GCN import GCNNet
from dataloader import load_dataset, load_data_GEOM
import json
import argparse
from torch_geometric.data import Data
if __name__ == '__main__':


    print("begin single train for gcn")

    """parse configuration for experiments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='indicate the source of hyperparameter configuration')
    parser.add_argument('--configs',type=json.loads, help='hyperparameter configuration for one single job')
    args = parser.parse_args()

    if args.source == 'fixed':
        """load configuration for experiments"""
        configs = load_experiment_configs("configs/gcn.fixed.yaml")

    else:
        """load configuration from command line"""
        configs = args.configs


    """reproducebale"""
    torch.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])

    """load the dataset and define the device: GPU or CPU"""

    # load dataset from pytorch geom
    # dataset = load_dataset(configs)
    # print(dataset)

    dataset = load_data_GEOM('Cornell/cornell', splits_file_path=None, train_percentage=0.2, val_percentage=0.2, embedding_mode=0.2,
              embedding_method=None,
              embedding_method_graph=None, embedding_method_space=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"running on: {device}")

    if configs['datasource'] == 'pygeo':
        model_gcn = GCNNet(dataset.num_node_features, dataset.num_classes, configs).to(device)

        path_for_saving_model = configs['path']+f"/gcn_{configs['dataset_name']}_{2}.pt"

        gcn_trainer = Trainer(model_gcn, configs, path_for_saving_model)

        # data = dataset[0].to(device)
        data = dataset.data.to(device)
        if configs["soft"]: data.y = torch.nn.functional.one_hot(data.y, dataset.num_classes)

        gcn_trainer.run(data, data.train_mask, data.y)
        print('test acc:')
        gcn_trainer.test_acc(data, data.y, True)
        gcn_trainer.records.update({'final_acc':gcn_trainer.test_acc(data, data.y, False)})
        print(f'best acc: {gcn_trainer.give_best_test()}')

        result_file = open(configs['path']+"/results_cornel.txt",'a')
        result_file.write(str(gcn_trainer.test_acc(data, data.y, False))+'\n')
        result_file.close()
        """ store the records"""
        path_for_records = configs['path'] + f"/gcn_{configs['dataset_name']}_{configs['count']}.records"
        gcn_trainer.save(path_for_records)




