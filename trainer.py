#%%
# Imports
from torch import torch, nn
from tqdm import tqdm

class Trainer():
    def __init__(self, device, model, train_loader, val_loader, optimizer, loss_fcn):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fcn = loss_fcn
    
    def compute_loss(self, batch, train, labels):
        self.model.validate()

        print('the batch size:', batch.size())
        print('the label size:', labels.size())

        loss = 0


        return loss
    def make_prediction(self, output):
        activation = torch.nn.Sigmoid()
        preds = torch.argmax( activation(output), 1)
        return preds
        # return torch.argmax(output, 1)
    
    def compute_acc(self, labels, output):
        #this will compute the accuracy of one batch
        batch_size = output.size(0)
        prediction = self.make_prediction(output)
        # print('predictions:', prediction)
        # print('regular output:', output)
        correct = (prediction.eq(labels)).sum().double()
        acc = correct / batch_size
        # print('correct:', correct)
        # print('predictions:', prediction)
        # print('labels:', labels)
        return acc
    
    def train_epoch(self, save_model = False):
        self.model.train()
        # if batch_size is None:
        #     batch_size = len(self.train_loader)
        # else:
        #     batch_size = min(batch_size, len(self.train_loader))

        # print('data loader size', len(self.train_loader['label']))

        # for i, sample in enumerate(self.train_loader):
        #     print(sample['label'], len(sample['label']))

        loss_list = []
        with tqdm(enumerate(self.train_loader), 
                  total=len(self.train_loader), 
                  desc='train epochs') as progress_bar:
            for i_batch, batch in progress_bar:
                data = batch['data'].to(self.device)

                # print('data shape:', data.size())
                # print(data.type)
                # print('labels:', batch['label'])
                labels = batch['label'].to(self.device)

                output = self.model(data)

                acc = self.compute_acc(labels=labels,
                                       output=output)
                # print('accuracy:', acc.item())

                #zero out the parameter gradients
                self.optimizer.zero_grad()
                # print('output', output.size())
                # print('labels', labels.size())

                loss = self.loss_fcn(output, labels)
                # print('loss:', loss.item())
                
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss)
                # progress_bar.set_postfix(avg_loss=sum(loss_list)/len(loss_list))
                progress_bar.set_postfix(loss = loss.item(), acc = acc.item())
        
        if save_model:
            torch.save(self.model.state_dict(), self.model.path)

        return loss_list

    def validate(self, sample_size = None):
        self.model.eval()
        if sample_size is None:
            sample_size = len(self.val_loader)
        else:
            sample_size = min(sample_size, len(self.val_loader))
        loss_list = []
        with tqdm(enumerate(self.val_loader), 
                  total=len(self.val_loader), 
                  desc='val losses') as progress_bar:
            for i, batch in progress_bar:
                if i == sample_size:
                    break

                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)

                out = self.model(data)
                l = self.loss_fcn(out, labels)
                loss_list.append(l)
                
                progress_bar.set_postfix(avgloss=(sum(loss_list)/len(loss_list)).item())

        return loss_list
#%%


#%%
