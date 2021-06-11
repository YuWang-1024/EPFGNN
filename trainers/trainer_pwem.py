from .trainer import Trainer
import torch
from early_stoppings import PWEMEarlyStoppingCriterionForM,PWEMEarlyStoppingCriterionForE,PWEMEarlyStoppingCriterion

class Trainer_PWEM(Trainer):
    def __init__(self, model, configs, path):
        super().__init__(model, configs, path)
        self.records = {"eval_acc": torch.tensor([]),
                        "test_acc": torch.tensor([]),
                        "train_loss": torch.tensor([]),
                        "eval_loss": torch.tensor([]),
                        "configs": configs}

        self.train_mode = configs['train_mode']


    def train(self, data, target, deg, configs, num_m_steps, num_e_steps):

        torch.save(data.y, self.path + 'q.tensor')
        torch.save(self.model.state_dict(), self.path)
        torch.save(self.optimizer.state_dict(), self.path + '.opt')

        if configs['stopping_name'] == 'pwemStoppingCriterionForEM':
            self.early_stopping_for_M = PWEMEarlyStoppingCriterionForM(patience=50, path=self.path, init_best_val_acc=configs['init_best_val_acc'])
            self.early_stopping_for_E = PWEMEarlyStoppingCriterionForE(patience=5, path=self.path, init_best_val_acc=configs['init_best_val_acc'])


            for i in range(configs['num_EM_steps']):
                loss, val_loss, val_acc, test_acc = self.run_with_early_stopping_in_EM(data, target, deg, num_m_steps, num_e_steps)

                print(f"number of EM step: {i}, train_loss: {loss}, val_loss: {val_loss}, val_acc: {val_acc}, test_acc: {test_acc}")
                if self.early_stopping_criterion.should_stop(epoch=i, val_accuracy=val_acc, model=self.model,
                                                             optimizer=self.optimizer, q=data.y):
                    return self.early_stopping_criterion.after_stopping_ops(self.model, self.optimizer)

        else:
            for i in range(configs['num_EM_steps']):
                loss, val_loss, val_acc, test_acc = self.run_without_EM(data, target, deg)
                print(f"number of EM step: {i}, train_loss: {loss}, val_loss: {val_loss}, val_acc: {val_acc}, test_acc: {test_acc}")
                if self.early_stopping_criterion.should_stop(epoch=i, val_accuracy=val_acc, model=self.model, optimizer=self.optimizer, q=data.y):
                    return self.early_stopping_criterion.after_stopping_ops()


    def run(self, data, target, deg, num_m_steps, num_e_steps):
        for i in range(num_m_steps):
            if self.train_mode == "joint training":
                loss = self.update_j(data)
                val_loss = self.evaluate_loss_j(data, data.val_mask)
            else:
                loss = self.update(data, deg)
                val_loss = self.evaluate_loss(data, deg, data.val_mask)

            self.records['train_loss'] = torch.cat((self.records['train_loss'], torch.tensor([loss])), dim=0)
            self.records['eval_loss'] = torch.cat((self.records['eval_loss'], torch.tensor([val_loss])), dim=0)

        for i in range(num_e_steps):
            if self.configs['mean_field'] == False:

                with torch.no_grad(): # only use node representation to do the inference test
                    val_acc = self.evaluate_acc(torch.softmax(self.model.gnnbackbone(data), dim=-1), target, data.val_mask, self.configs['verbose'])
                    test_acc = self.evaluate_acc(torch.softmax(self.model.gnnbackbone(data), dim=-1), target, data.test_mask, self.configs['verbose'])
            else:
                with torch.no_grad():
                    data.y[~data.train_mask] = self.model.inference(data)[~data.train_mask]
                    val_acc = self.evaluate_acc(data.y, target, data.val_mask, self.configs['verbose'])
                    test_acc = self.evaluate_acc(data.y, target, data.test_mask, self.configs['verbose'])
            self.records['eval_acc'] = torch.cat((self.records['eval_acc'], torch.tensor([val_acc])), dim=0)
            self.records['test_acc'] = torch.cat((self.records['test_acc'], torch.tensor([test_acc])), dim=0)

        return loss, val_loss, val_acc, test_acc

    def run_with_early_stopping_in_EM(self, data, target, deg, num_m_steps, num_e_steps):
        print("run_with_early_stopping_in_EM")
        pre_trained_steps = self.configs['pre_trained_steps']

        for i in range(num_m_steps):
            self.epoch = pre_trained_steps+i
            self.model.train()

            if self.train_mode == "pre-training":
                loss = self.update(data, deg)
                val_loss = self.evaluate_loss(data, deg, data.val_mask)
            elif self.train_mode == "center_redist":
                loss = self.update_redist2(data,deg)
                val_loss = self.evaluate_loss_redist2(data, deg, data.val_mask)
            elif self.train_mode == "joint_redist":
                loss = self.update_redist2(data, deg)
                val_loss = self.evaluate_loss_redist2(data, deg, data.val_mask)
            elif self.train_mode == "center_edge_rezero":
                loss = self.update_redist2_edge_rezero(data,deg)
                val_loss = self.evaluate_loss_edge_rezero(data, deg, data.val_mask)
            elif self.train_mode == "average_edge_rezero":
                loss = self.update_edge_rezero(data, deg)
                val_loss = self.evaluate_loss_average_edge_rezero(data, deg, data.val_mask)


            self.records['train_loss'] = torch.cat((self.records['train_loss'], torch.tensor([loss])), dim=0)
            self.records['eval_loss'] = torch.cat((self.records['eval_loss'], torch.tensor([val_loss])), dim=0)

            test_temp_q = self.model.inference(data, deg)
            val_acc = self.evaluate_acc(test_temp_q, target, data.val_mask, self.configs['verbose'])
            test_acc = self.evaluate_acc(test_temp_q, target, data.test_mask, self.configs['verbose'])
            if self.early_stopping_for_M.should_stop(epoch=i, val_accuracy=val_acc, model=self.model,
                                                     optimizer=self.optimizer, q=data.y):
                self.configs['pre_trained_steps'] = pre_trained_steps+i+1
                break

        self.early_stopping_for_M.count = 0
        self.model.load_state_dict(torch.load(self.path))
        self.optimizer.load_state_dict(torch.load(self.path + '.opt'))


        for i in range(num_e_steps):
            self.model.eval()
            if self.configs['mean_field'] == False:
                with torch.no_grad(): # only use node representation to do the inference test
                    val_acc = self.evaluate_acc(torch.softmax(self.model.gnnbackbone(data), dim=-1), target, data.val_mask, self.configs['verbose'])
                    test_acc = self.evaluate_acc(torch.softmax(self.model.gnnbackbone(data), dim=-1), target, data.test_mask, self.configs['verbose'])
            else:
                with torch.no_grad():
                    data.y[~data.train_mask] = self.model.inference(data, deg)[~data.train_mask]
                    val_acc = self.evaluate_acc(data.y, target, data.val_mask, self.configs['verbose'])
                    test_acc = self.evaluate_acc(data.y, target, data.test_mask, self.configs['verbose'])

            self.records['eval_acc'] = torch.cat((self.records['eval_acc'], torch.tensor([val_acc])), dim=0)
            self.records['test_acc'] = torch.cat((self.records['test_acc'], torch.tensor([test_acc])), dim=0)
            if self.early_stopping_for_E.should_stop(epoch=i, val_accuracy=val_acc, q=data.y):break


        self.early_stopping_for_E.count = 0
        data.y[~data.train_mask] = self.early_stopping_for_E.after_stopping_ops()[~data.train_mask]
        val_acc = self.evaluate_acc(data.y, target, data.val_mask, self.configs['verbose'])
        test_acc = self.evaluate_acc(data.y, target, data.test_mask, self.configs['verbose'])

        return loss, val_loss, val_acc, test_acc



    def update(self, data, deg): # update function for pre-training
        # m step function
        self.model.train()
        self.optimizer.zero_grad()

        U, U_redistributed, B, B_redistributed, Z = self.model(data, deg)
        loss_for_pieces = self.criterion(U_redistributed, data.edge_index, B_redistributed, Z, data.y)
        loss = -torch.mean(loss_for_pieces)
        loss.backward()
        self.optimizer.step()

        return loss


    def update_use_train_data(self, data, deg): #update function for pre-pwem
        self.model.train()
        self.optimizer.zero_grad()
        U, U_redistributed, B, B_redistributed, Z = self.model(data, deg)
        loss_for_pieces = self.criterion(U_redistributed, data.edge_index, B_redistributed, Z, data.y)
        loss = -torch.mean(loss_for_pieces[data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss

    def update_redist2(self, data, deg):
        self.model.train()
        self.optimizer.zero_grad()
        U, B, Z = self.model(data, deg)
        loss_for_pieces = self.criterion(U, data.edge_index, B/2, Z, data.y)
        loss = -torch.mean(loss_for_pieces)
        loss.backward()
        self.optimizer.step()
        return loss


    def update_redist2_edge_rezero(self, data,deg):
        self.model.train()
        self.optimizer.zero_grad()
        U, B, Z, R= self.model(data, deg)
        loss_for_pieces = self.criterion(U, data.edge_index, B/2, Z, data.y, R)
        loss = -torch.mean(loss_for_pieces)
        loss.backward()
        self.optimizer.step()

        return loss

    def update_edge_rezero(self, data,deg):
        self.model.train()
        self.optimizer.zero_grad()
        U, U_redistribution, B, B_redistribution, Z, R= self.model(data, deg)
        loss_for_pieces = self.criterion(U_redistribution, data.edge_index, B_redistribution, Z, data.y, R)
        loss = -torch.mean(loss_for_pieces)
        loss.backward()
        self.optimizer.step()
        return loss


    def evaluate_loss(self, data, deg, mask):
        self.model.eval()
        with torch.no_grad():
            U, U_redistributed, B, B_redistributed, Z = self.model(data, deg)
            loss_for_pieces = self.criterion(U_redistributed, data.edge_index, B_redistributed, Z, data.y)
            loss = -torch.mean(loss_for_pieces[mask])
            return loss

    def evaluate_loss_redist2(self, data, deg,mask):
        self.model.eval()
        with torch.no_grad():
            U, B, log_z_redistributed = self.model(data,deg)
            loss_for_pieces = self.criterion(U, data.edge_index, B/2, log_z_redistributed,
                                             data.y)
            loss = -torch.mean(loss_for_pieces[mask])
            return loss

    def evaluate_loss_edge_rezero(self, data, deg, mask):
        self.model.eval()
        with torch.no_grad():
            U, B, log_z_redistributed, R = self.model(data,deg)
            loss_for_pieces = self.criterion(U, data.edge_index, B/2, log_z_redistributed,
                                             data.y, R)
            loss = -torch.mean(loss_for_pieces[mask])
            return loss

    def evaluate_loss_average_edge_rezero(self, data, deg, mask):
        self.model.eval()
        with torch.no_grad():
            U, U_redistribution, B, B_redistribution, log_z_redistributed, R = self.model(data,deg)
            loss_for_pieces = self.criterion(U_redistribution, data.edge_index, B_redistribution, log_z_redistributed,
                                             data.y, R)
            loss = -torch.mean(loss_for_pieces[mask])
            return loss

    def evaluate_acc(self, input, target, mask, verbose):
        with torch.no_grad():
            acc = self.metric(input, target, mask, verbose)
        return acc

    def test_acc(self, input, target, mask, verbose):
        with torch.no_grad():
            acc = self.metric(input, target, mask, verbose)
        return acc

