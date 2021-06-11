import torch
from utils import load_experiment_configs
from trainers.trainer_pwem import Trainer_PWEM
from trainers.trainer import Trainer
from models.GCN import GCNNet
from models.GAT import GATNet
from dataloader import load_dataset, load_data_GEOM
import json
import argparse
from torch_geometric.data import Data
from models.PWEM import PWGNN
from torch_geometric.utils import degree

if __name__ == '__main__':

    print("begin single train for pwem")

    """parse configuration for experiments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='indicate the source of hyperparameter configuration')
    parser.add_argument('--configs', type=json.loads, help='hyperparameter configuration for one single job')
    args = parser.parse_args()
    if args.source == 'fixed':
        """load configuration for experiments"""
        print("load from fixed yaml file")
        configs = load_experiment_configs("configs/pwem.fixed.yaml")

    else:
        """load configuration from command line"""
        configs = args.configs
    print(configs['path'])

    """reproducebale"""
    torch.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])

    """load the dataset and define the device: GPU or CPU"""
    # dataset = load_dataset(configs)
    dataset = load_data_GEOM('squirrel', splits_file_path=None, train_percentage=0.2, val_percentage=0.2,
                             embedding_mode=0.2,
                             embedding_method=None,
                             embedding_method_graph=None, embedding_method_space=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = dataset[0].to(device)
    data = dataset.data.to(device)

    print(f"running on: {device}")

    print(dataset)
    print(configs['backbone_name'])

    if configs['datasource'] == 'pygeo':

        if configs['backbone_name'] == 'gcn':
            model_q = GCNNet(dataset.num_node_features, int(dataset.num_classes), configs).to(device)
        elif configs['backbone_name'] == 'gat':
            model_q = GATNet(dataset.num_node_features, int(dataset.num_classes), configs).to(device)

        if configs["soft"]:
            data.y = torch.nn.functional.one_hot(data.y.to(torch.long), int(dataset.num_classes))


        """initialization of model q"""
        path_for_model_q = configs['path'] + f"/pwem_model_q_{configs['dataset_name']}_{configs['count']}.pt"
        configs['criterion_name'] ='crossEntropy'
        configs['stopping_name'] = 'gcnStoppingCriterion'
        trainer_q = Trainer(model_q, configs, path_for_model_q)
        pretrain_test_acc=0

        if configs['train_mode'] == 'pre-training' \
                or configs['train_mode'] =='center_redist' \
                or configs['train_mode'] == 'center_edge_rezero' \
                or configs['train_mode'] == 'average_edge_rezero' :
            """pre-train procedure"""

            print("pre-trainning procedure")
            trainer_q.run(data, data.train_mask, data.y)
            pretrain_test_acc = trainer_q.test_acc(data, data.y, False)
            val_acc = trainer_q.evaluate_acc(data, data.y, False)
            print(f"val acc for pretrain: {val_acc}")
            print(f"test acc for pretrain: {pretrain_test_acc}")
            print(f"best acc: {trainer_q.give_best_test()}")
            configs["init_best_val_acc"] = val_acc


        """EM training procedure"""
        print("EM trianing procedure")

        """initialization of model p"""
        inputs_p = data.x
        targets_p = torch.zeros(data.x.shape[0],dataset.num_classes).to(device)
        data_p = Data(edge_index=data.edge_index, test_mask=data.test_mask, train_mask=data.train_mask, val_mask=data.val_mask, x=inputs_p, y=targets_p)

        if configs['train_mode'] != 'joint training':
            q = trainer_q.predict(data)
            data_p.y = q
        data_p.y[data.train_mask] = data.y[data.train_mask].to(torch.float)

        # for 2 step training, use gold is in reverse degrade the performance.

        if configs['fixed'] == True:
            print("set the params in model q requires grade equals false")
            for p in model_q.parameters():
                p.requires_grad = False


        elif configs['train_mode']=='pre-training':
            model_p = PWGNN(model_q, configs).to(device)
            configs["criterion_name"] = 'pwLoss'

        elif configs['train_mode'] =='center_redist':
            model_p = PWGNN(model_q, configs).to(device)
            configs["criterion_name"] = 'PWLoss_redis2'

        elif configs['train_mode'] == 'joint_redist':
            model_p  = PWGNN(model_q, configs).to(device)
            configs["criterion_name"] = 'PWLoss_redis2'

        elif configs['train_mode'] == 'center_edge_rezero':
            configs['rezero_size'] = data.edge_index.shape[1]
            print(configs['rezero_size'])
            model_p = PWGNN(model_q, configs).to(device)
            configs["criterion_name"] = 'PWLoss_redist2_edge_rezero'

        elif configs['train_mode'] == 'average_edge_rezero':
            configs['rezero_size'] = data.edge_index.shape[1]
            print(configs['rezero_size'])
            model_p = PWGNN(model_q, configs).to(device)
            configs['criterion_name'] = 'PWLoss_average_edge_rezero'


        """ change the weight decay for two procedure"""
        if configs["change_lr"] == True:
            configs["weight_decay"] = 5e-5
            configs["learning_rate"] = 0.01


        path_for_model_p = configs['path']+f"/pwem_model_p_{configs['dataset_name']}_{configs['count']}.pt"

        configs["stopping_name"] = 'pwemStoppingCriterionForEM'
        # configs["stopping_name"] = 'pwemStoppingCriterion'

        configs['patience'] = 10
        if configs['change_EMPatience']: configs['patience']=5

        trainer_pwem = Trainer_PWEM(model_p, configs, path_for_model_p)
        row,_ = data_p.edge_index
        deg = degree(row, data.x.size(0), dtype=data.x.dtype).view(-1, 1) + 1 # here already add self-loop

        num_m_steps = data.x.shape[0]//30 + 1
        num_e_steps = 10
        configs['num_m_steps'] = num_m_steps
        configs['num_e_steps'] = num_e_steps

        """call EM training precedure"""
        q = trainer_pwem.train(data_p, data.y, deg, configs, num_m_steps, num_e_steps)
        final_EM_test_acc = trainer_pwem.test_acc(data_p.y, data.y, data_p.test_mask, False)
        print(final_EM_test_acc)



        """write results into txt file"""
        # result_file = open(configs['path']+f"/results_{configs['model']}_{configs['dataset_name']}_{configs['variant']}.txt",'a')
        result_file = open(configs['path']+f"/results_squirrel_ninit.txt",'a')
        result_file.write(str(final_EM_test_acc)+'\n')
        result_file.close()

        # result_file = open(configs['path']+f"/results_{configs['model']}_{configs['dataset_name']}_{configs['variant']}_PRE.txt",'a')
        result_file = open(configs['path']+f"/results_squirrel_ninit_PRE.txt",'a')
        result_file.write(str(pretrain_test_acc)+'\n')
        result_file.close()

        # result_file = open(configs['path']+f"/results_{configs['model']}_{configs['dataset_name']}_{configs['variant']}_configs.txt",'a')
        result_file = open(configs['path']+f"/results_squirrel_ninit_configs.txt",'a')
        result_file.write(json.dumps(configs)+ '\n')
        result_file.close()
        """ store the records"""
        path_for_records_p = configs['path'] + f"/pwem_model_p_{configs['dataset_name']}_{configs['count']}.records"
        trainer_pwem.save(path_for_records_p)







