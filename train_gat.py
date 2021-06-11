import torch
from utils import load_experiment_configs
from trainers.trainer import Trainer
from models.GAT import GATNet
from dataloader import load_dataset
import json
import argparse

if __name__ == '__main__':

    print("begin single train for gat")

    """parse configuration for experiments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='indicate the source of hyperparameter configuration')
    parser.add_argument('--configs', type=json.loads, help='hyperparameter configuration for one single job')
    args = parser.parse_args()

    if args.source == 'fixed':
        """load configuration for experiments"""
        configs = load_experiment_configs("configs/gat.fixed.yaml")

    else:
        """load configuration from command line"""
        configs = args.configs

    """reproduceable"""
    torch.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])

    """load the dataset and define the device: GPU or CPU"""
    dataset = load_dataset(configs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"running on: {device}")


    if configs['datasource'] == 'pygeo':
        model_gat = GATNet(dataset.num_node_features, dataset.num_classes, configs).to(device)

        path_for_saving_model = configs['path'] + f"/gat_{configs['dataset_name']}_{configs['count']}.pt"

        gat_trainer = Trainer(model_gat, configs, path_for_saving_model)
        data = dataset[0].to(device)
        if configs["soft"]: data.y = torch.nn.functional.one_hot(data.y, dataset.num_classes)

        gat_trainer.run(data, data.train_mask, data.y)
        print('test acc:')
        gat_trainer.test_acc(data, data.y, True)
        gat_trainer.records.update({'final_acc':gat_trainer.test_acc(data, data.y, False)})
        print(f'best acc: {gat_trainer.give_best_test()}')


        result_file = open(configs['path']+"/results_pubmed_0128.txt",'a')
        result_file.write(str(gat_trainer.test_acc(data, data.y, False))+'\n')
        result_file.close()
        """ store the records"""
        path_for_records = configs['path'] + f"/gat_{configs['dataset_name']}_{configs['count']}.records"
        gat_trainer.save(path_for_records)




