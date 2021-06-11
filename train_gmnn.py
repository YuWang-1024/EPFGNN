import torch
from utils import load_experiment_configs
from trainers.trainer import Trainer
from models.GMNN import GMNNNet
from dataloader import load_dataset, load_data_GEOM
import json
import argparse
from torch_geometric.data import Data
from models.GCN import GCNNet
from models.GAT import GATNet

if __name__ == '__main__':

    print("begin single train for model gmnn")

    """parse configuration for experiments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='indicate the source of hyperparameter configuration')
    parser.add_argument('--configs', type=json.loads, help='hyperparameter configuration for one single job')
    args = parser.parse_args()

    if args.source == 'fixed':
        """load configuration for experiments"""
        configs = load_experiment_configs("configs/gmnn.fixed.yaml")

    else:
        """load configuration from command line"""
        configs = args.configs

    """reproducebale"""
    torch.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    
    
    """load the dataset and define the device: GPU or CPU"""
    # dataset = load_dataset(configs)
    dataset = load_data_GEOM('chameleon', splits_file_path=None, train_percentage=0.2, val_percentage=0.2, embedding_mode=0.2,
              embedding_method=None,
              embedding_method_graph=None, embedding_method_space=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = dataset[0].to(device)
    data = dataset.data.to(device)
    print(dataset)
    print(f"running on: {device}")

    
    """training procedure"""


    def init_q_data(data_q, data):
        data_q.x = data.x
        if configs['soft']:data_q.y[data.train_mask] = data.y[data.train_mask].float()
        else: data_q.y[data.train_mask] = data.y[data.train_mask]


    def update_q_data(trainer_p, data_p, data_q):
        preds = trainer_p.predict(data_p)
        data_q.y = preds
        if configs['use_gold']:
            if configs['soft']:data_q.y[data_q.train_mask] = data_q.gold[data_q.train_mask].float()
            else:data_q.y[data_q.train_mask] = data_q.gold[data_q.train_mask]


    def update_p_data(trainer_q, data_q, data_p):
        preds = trainer_q.predict(data_q)
        if configs['draw'] == 'exp':
            data_p.x = preds
            data_p.y = preds

        elif configs['draw'] == 'max':
            idx_lb = torch.max(preds, dim=-1)[1]
            data_p.x.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            data_p.y.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)

        elif configs['draw'] == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            data_p.x.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            data_p.y.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)

        if configs['use_gold']:
            if configs['soft']:
                data_p.x[data_p.train_mask] = data_p.gold[data_p.train_mask].float()
                data_p.y[data_p.train_mask] = data_p.gold[data_p.train_mask].float()


    if configs['datasource'] == 'pygeo':

        if configs['backbone_name']=='gcn':
            model_gmnn = GMNNNet(GCNNet, dataset.num_node_features, dataset.num_classes, configs).to(device)
        elif configs['backbone_name']=='gat':
            model_gmnn = GMNNNet(GATNet, dataset.num_node_features, dataset.num_classes, configs).to(device)

        path_for_model_q = configs['path'] + f"/gmnn_model_q_{configs['dataset_name']}_{configs['count']}.pt"
        path_for_model_p = configs['path'] + f"/gmnn_model_p_{configs['dataset_name']}_{configs['count']}.pt"

        modelq_configs = configs.copy()
        modelq_configs['tensorboard'] = 'model_q'
        trainer_q = Trainer(model_gmnn.GNN_q, modelq_configs, path_for_model_q)

        modelp_configs = configs.copy()
        modelp_configs['tensorboard'] = 'model_p'
        trainer_p = Trainer(model_gmnn.GNN_p, modelp_configs, path_for_model_p)


        if configs['soft']: data.y = torch.nn.functional.one_hot(data.y, dataset.num_classes)

        inputs_q = torch.zeros(data.x.shape[0], dataset.num_features).to(device)  # [N,F]
        target_q = torch.zeros(data.x.shape[0], dataset.num_classes).to(device)  # [N, C]
        inputs_p = torch.zeros(data.x.shape[0], dataset.num_classes).to(device)  # [N, C]
        target_p = torch.zeros(data.x.shape[0], dataset.num_classes).to(device)  # [N, C]

        data_q = Data(edge_index=data.edge_index, test_mask=data.test_mask, train_mask=data.train_mask, val_mask=data.val_mask, x=inputs_q, y=target_q, gold=data.y).to(device)
        data_p = Data(edge_index=data.edge_index, test_mask=data.test_mask, train_mask=data.train_mask, val_mask=data.val_mask, x=inputs_p, y=target_p, gold=data.y).to(device)

        """pre train procedure"""
        init_q_data(data_q,data)
        trainer_q.run(data_q, data.train_mask, data_q.gold)
        final_acc_pretrain = trainer_q.test_acc(data, data.y, False)
        print(f"pre-train test: {final_acc_pretrain}")
        trainer_q.records.update({'final_acc_pretrain': final_acc_pretrain})

        # need to clear the pretrained records
        # trainer_q.records["eval_acc"] = torch.tensor([])

        """EM train procedure"""
        all_mask = torch.ones_like(data.train_mask).to(device)
        num_EM_steps = configs['num_EM_steps']
        for step in range(num_EM_steps):

            # print("m-step/model_p")
            update_p_data(trainer_q, data_q, data_p)
            trainer_p.run(data_p, all_mask, data_p.gold)


            # print("e-step/model_q")
            update_q_data(trainer_p, data_p, data_q)
            trainer_q.run(data_q, all_mask, data_q.gold)


        final_EM_test_acc = trainer_q.test_acc(data, data.y,False)
        print(f"final test {final_EM_test_acc}")
        print(f"best test: {trainer_q.give_best_test()}")
        trainer_q.records.update({'final_EM_test_acc': final_EM_test_acc})


        result_file = open(configs['path']+"/results_chameleon.txt",'a')
        result_file.write(str(trainer_q.test_acc(data, data.y, False))+'\n')
        result_file.close()
        """ store the records"""
        path_for_records_q = configs['path'] + f"/gmnn_model_q_{configs['dataset_name']}_{configs['count']}.records"
        # path_for_records_p = configs['path'] + f"/gmnn_model_p_{configs['dataset_name']}_{configs['count']}.records"
        trainer_q.save(path_for_records_q)
        # trainer_p.save(path_for_records_p)





