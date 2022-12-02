import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_training_data import load_data, load_tensors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
import os
from plots import plot_learning_curves
import json

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class LSTM(nn.Module):

    def __init__(self, embedding_dim):
        super(LSTM, self).__init__()
        hidden_dim = 64
        hidden_dim2 = 2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        # Outcomes: abdominal, advanced-cad, alcohol-abuse, asp-for-mi, creatinine, dietsupp-2mos
        #   drug-abuse, english, hba1c, keto-1yr, major-diabetes, makes-decisions, mi-6mos


        ##### new
        # self.abdominal = nn.RNN(hidden_dim, 2)
        # self.advancedcad = nn.RNN(hidden_dim, 2)
        # self.alcoholabuse = nn.RNN(hidden_dim, 2)
        # self.aspformi = nn.RNN(hidden_dim, 2)
        # self.creatinine = nn.RNN(hidden_dim, 2)
        # self.dietsupp = nn.RNN(hidden_dim, 2)
        # self.drugabuse = nn.RNN(hidden_dim, 2)
        # self.english = nn.RNN(hidden_dim, 2)
        # self.hba1c = nn.RNN(hidden_dim, 2)
        # self.keto = nn.RNN(hidden_dim, 2)
        # self.diabetes = nn.RNN(hidden_dim, 2)
        # self.decisions = nn.RNN(hidden_dim, 2)
        # self.mi = nn.RNN(hidden_dim, 2)


        self.abdominal = nn.RNN(hidden_dim, hidden_dim2)
        self.advancedcad = nn.RNN(hidden_dim, hidden_dim2)
        self.alcoholabuse = nn.RNN(hidden_dim, hidden_dim2)
        self.aspformi = nn.RNN(hidden_dim, hidden_dim2)
        self.creatinine = nn.RNN(hidden_dim, hidden_dim2)
        self.dietsupp = nn.RNN(hidden_dim, hidden_dim2)
        self.drugabuse = nn.RNN(hidden_dim, hidden_dim2)
        self.english = nn.RNN(hidden_dim, hidden_dim2)
        self.hba1c = nn.RNN(hidden_dim, hidden_dim2)
        self.keto = nn.RNN(hidden_dim, hidden_dim2)
        self.diabetes = nn.RNN(hidden_dim, hidden_dim2)
        self.decisions = nn.RNN(hidden_dim, hidden_dim2)
        self.mi = nn.RNN(hidden_dim, hidden_dim2)

        # self.abdominal_fc = nn.Linear(hidden_dim2, 2)
        # self.advancedcad_fc = nn.Linear(hidden_dim2, 2)
        # self.alcoholabuse_fc = nn.Linear(hidden_dim2, 2)
        # self.aspformi_fc = nn.Linear(hidden_dim2, 2)
        # self.creatinine_fc = nn.Linear(hidden_dim2, 2)
        # self.dietsupp_fc = nn.Linear(hidden_dim2, 2)
        # self.drugabuse_fc = nn.Linear(hidden_dim2, 2)
        # self.english_fc = nn.Linear(hidden_dim2, 2)
        # self.hba1c_fc = nn.Linear(hidden_dim2, 2)
        # self.keto_fc = nn.Linear(hidden_dim2, 2)
        # self.diabetes_fc = nn.Linear(hidden_dim2, 2)
        # self.decisions_fc = nn.Linear(hidden_dim2, 2)
        # self.mi_fc = nn.Linear(hidden_dim2, 2)

    def forward(self, input, lengths):
        sigmoid_fun = nn.Sigmoid()
        relu_fun = nn.ReLU()

        batch_size, seq_len, feature_len = input.size()
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed_input)
        

        ###### New
        inter_dict = {'abdominal': (self.abdominal(output)), 'advanced-cad': (self.advancedcad(output)), 'alcohol-abuse': (self.alcoholabuse(output)), 
                    'asp-for-mi': (self.aspformi(output)), 'creatinine': (self.creatinine(output)), 'dietsupp-2mos': (self.dietsupp(output)), 
                    'drug-abuse': (self.drugabuse(output)), 'english': (self.english(output)), 'hba1c': (self.hba1c(output)), 
                    'keto-1yr': (self.keto(output)), 'major-diabetes': (self.diabetes(output)), 'makes-decisions': (self.decisions(output)), 
                    'mi-6mos': (self.mi(output))}
        
        # inter_dict2 = {'abdominal': self.abdominal_fc, 'advanced-cad': self.advancedcad_fc, 'alcohol-abuse': self.alcoholabuse_fc, 
        #             'asp-for-mi': self.aspformi_fc, 'creatinine': self.creatinine_fc, 'dietsupp-2mos': self.dietsupp_fc, 
        #             'drug-abuse': self.drugabuse_fc, 'english': self.english_fc, 'hba1c': self.hba1c_fc, 
        #             'keto-1yr': self.keto_fc, 'major-diabetes': self.diabetes_fc, 'makes-decisions': self.decisions_fc, 
        #             'mi-6mos': self.mi_fc}
        final_dict = {}

        #### old
        for k,v in inter_dict.items():
            padded_output, lengths = torch.nn.utils.rnn.pad_packed_sequence(v[0], batch_first=False, total_length=seq_len)
            padded_output = padded_output.view(batch_size*seq_len, 2)
            adjusted_lengths = [(l-1)*batch_size + i for i,l in enumerate(lengths)]
            lengthTensor = torch.tensor(adjusted_lengths, dtype=torch.int64)
            padded_output = padded_output.index_select(0,lengthTensor)
            padded_output = padded_output.view(batch_size,2)
            final_dict[k] = sigmoid_fun(padded_output)

        return final_dict
        # ######

        # for k,v in inter_dict.items():
        #     padded_output, lengths = torch.nn.utils.rnn.pad_packed_sequence(v[0], batch_first=False, total_length=seq_len)
        #     padded_output = padded_output.view(batch_size*seq_len, 30)
        #     adjusted_lengths = [(l-1)*batch_size + i for i,l in enumerate(lengths)]
        #     lengthTensor = torch.tensor(adjusted_lengths, dtype=torch.int64)
        #     padded_output = padded_output.index_select(0,lengthTensor)
        #     padded_output = padded_output.view(batch_size,30)

        #     activated_output = relu_fun(padded_output)
        #     final_dict[k] = sigmoid_fun(inter_dict2[k](activated_output))
        #     # final_dict[k] = sigmoid_fun(padded_output)

        return final_dict
        #

    def get_loss(self, net_output, ground_truth, labels, criterion):
        abdominal_loss = criterion(net_output['abdominal'], ground_truth[:,labels.index('abdominal')])
        advancedcad_loss = criterion(net_output['advanced-cad'], ground_truth[:,labels.index('advanced-cad')])
        alcoholabuse_loss = criterion(net_output['alcohol-abuse'], ground_truth[:,labels.index('alcohol-abuse')])
        aspformi_loss = criterion(net_output['asp-for-mi'], ground_truth[:,labels.index('asp-for-mi')])
        creatinine_loss = criterion(net_output['creatinine'], ground_truth[:,labels.index('creatinine')])
        dietsupp_loss = criterion(net_output['dietsupp-2mos'], ground_truth[:,labels.index('dietsupp-2mos')])
        drugabuse_loss = criterion(net_output['drug-abuse'], ground_truth[:,labels.index('drug-abuse')])
        english_loss = criterion(net_output['english'], ground_truth[:,labels.index('english')])
        hba1c_loss = criterion(net_output['hba1c'], ground_truth[:,labels.index('hba1c')])
        keto_loss = criterion(net_output['keto-1yr'], ground_truth[:,labels.index('keto-1yr')])
        diabetes_loss = criterion(net_output['major-diabetes'], ground_truth[:,labels.index('major-diabetes')])
        decisions_loss = criterion(net_output['makes-decisions'], ground_truth[:,labels.index('makes-decisions')])
        mi_loss = criterion(net_output['mi-6mos'], ground_truth[:,labels.index('mi-6mos')])
        loss = abdominal_loss + advancedcad_loss + alcoholabuse_loss + aspformi_loss + aspformi_loss + creatinine_loss + \
            dietsupp_loss + drugabuse_loss + english_loss + hba1c_loss + keto_loss + diabetes_loss + decisions_loss + mi_loss

        return loss, {'abdominal': abdominal_loss, 'advanced-cad': advancedcad_loss, 'alcohol-abuse': alcoholabuse_loss, 'asp-for-mi': aspformi_loss, 'creatinine': creatinine_loss,  
        'dietsupp-2mos': dietsupp_loss, 'drug-abuse': drugabuse_loss, 'english': english_loss, 'hba1c': hba1c_loss, 'keto-1yr': keto_loss, 'major-diabetes': diabetes_loss, 'makes-decisions': decisions_loss, 'mi-6mos': mi_loss}

def compute_batch_accuracy(outputs, targets, labels):
    with torch.no_grad():

        all_outputs_dict = {}
        eval_dict = {'accuracy': {}, 'precision': {}, 'recall': {}, 'f1': {}}
        eval_list = [accuracy_score, precision_score, recall_score, f1_score]
        accuracies = []

        for i, label in enumerate(labels):
            _, y_pred = outputs[label].max(1)
            y_true = targets[:,i]
            eval_dict = {'accuracy': {}, 'precision': {}, 'recall': {}, 'f1': {}}
            for eval_idx, key in enumerate(eval_dict.keys()):
                eval_dict[key] = eval_list[eval_idx](y_true, y_pred)
            all_outputs_dict[label] = eval_dict.copy()
            accuracies.append(eval_dict['f1'])
        
        #returns mean f1 score
        mean_acc = sum(accuracies)/13
        return mean_acc, all_outputs_dict

def train_model(model, device, train_loader, criterion, optimizer, epoch, labels):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - end)
        
        # get the inputs
        inputs, targets, lengths = data
        
        input = inputs.to(device)
        target = targets.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input, lengths)

                
        loss_train, losses_train = model.get_loss(outputs, target, labels, criterion)
        mean_accuracy_train, accuracies_train = compute_batch_accuracy(outputs, target, labels)
        
        # total_loss_list.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss_train.item(), targets.size(0))
        accuracy.update(mean_accuracy_train.item(), targets.size(0))

    return losses.avg, accuracy.avg

def evaluate_model(model, device, data_loader, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    results = []

    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, data in enumerate(data_loader):
            input, target, lengths = data
            
            input = input.to(device)
            target = target.to(device)

            outputs = model(input, lengths)
            loss_validation, losses_validation = model.get_loss(outputs, target, labels, criterion)
            mean_accuracy_validation, accuracies_validation = compute_batch_accuracy(outputs, target, labels)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss_validation.item(), target.size(0))
            accuracy.update(mean_accuracy_validation, target.size(0))


    # return losses.avg, accuracy.avg, results
    return losses.avg, accuracy.avg

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    NUM_EPOCHS = 5
    BATCH_SIZE = 4
    USE_CUDA = True  # Set 'True' if you want to use GPU
    NUM_WORKERS = 2

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = LSTM(embedding_dim = 100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02 )


    training_data_path = 'training_data.json'
    x,y,lengths = load_data(training_data_path, "training_data")
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(x,y,lengths, test_size=0.2, random_state=1)

    train_tensor, labels = load_tensors(X_train,y_train,len_train)
    validation_tensor, _ = load_tensors(X_test,y_test,len_test)

    trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=10, shuffle=False)
    validationloader = torch.utils.data.DataLoader(dataset=validation_tensor, batch_size=BATCH_SIZE, shuffle=False)

    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    model.to(device)
    criterion.to(device)
    best_val_acc = 0

    for epoch in range(1, NUM_EPOCHS):
        train_loss, train_accuracy = train_model(model, device, trainloader, criterion, optimizer, epoch, labels)
        valid_loss, valid_accuracy = evaluate_model(model, device, validationloader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy.item())

        is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best f1-score
        if is_best:
            best_val_acc = valid_accuracy
            print(best_val_acc)
            torch.save(model, "LSTM_RNN_test.pth")
        
    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

    #### test model
    test_model = torch.load("LSTM_RNN_test.pth")
    testing_data_path = 'testing_data.json'
    x_test,y_test,len_test = load_data(testing_data_path, "testing_data")
    test_dataset, labels = load_tensors(x_test,y_test,len_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    test_model.eval()
    with torch.no_grad():
        for data in test_loader:
            # get the inputs
            inputs, targets, lengths = data

            if device.type == "cuda":
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = test_model(inputs, lengths)
            mean_test_accuracy, test_accuracies_dict = compute_batch_accuracy(outputs, targets, labels)
    print(mean_test_accuracy)
    print(test_accuracies_dict)


    


 
