from typing import Mapping, Union, Optional, Callable, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm, trange
import time
import matplotlib.pyplot as plt


# reproducibility stuff

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
torch.backends.cudnn.benchmark = False


np.set_printoptions(threshold=20000, linewidth=350, formatter={"str_kind": lambda x: x})

if os.name == "nt":
    CLEAR_STR = "cls" 
else:
    CLEAR_STR = "clear"
os.system(CLEAR_STR)



def run(model, dim, iters, sleep_time=0.2, mark="•", manual_mode=False):
    n = torch.randint(low=0, high=2, size=(1,1,dim,dim), dtype=torch.float)

    _print_board(n[0,0,...], mark=mark)
    print(f"\nStep number: {0}")
    print(f"\nPad mode: {pad_mode}")

    for i in range(iters):
        if manual_mode:
            input()
        n = torch.round(model(n))
        _print_board(n[0,0,...], mark=mark)
        print(f"\nStep number: {i+1}")
        print(f"\nPad mode: {pad_mode}")
        time.sleep(sleep_time)
        #print()
        #print(F.pad(n, (1,1,1,1), mode="circular"))
        print()



# Run GoL simulation with initial state init and transition given by "model"
def run_init(model, init, iters, sleep_time=0.2, mark="•", manual_mode=False):
    
    n = init

    _print_board(n[0,0,...], mark=mark)
    print(f"\nStep number: {0}")
    print(f"\nPad mode: {pad_mode}")

    for i in range(iters):
        if manual_mode:
            input()
        #print(model(n))
        #input()
        n = torch.round(model(n))
        _print_board(n[0,0,...], mark=mark)
        print(f"\nStep number: {i+1}")
        print(f"\nPad mode: {pad_mode}")
        time.sleep(sleep_time)
        #print()
        #print(F.pad(n, (1,1,1,1), mode="circular"))
        print()

    return n


# Fix the weights of a model to the correct one for the classic Game of Like rules
def gol_weights(model):
    params = list(model.parameters())
    pm = torch.tensor([[[[1, 1, 1],[1, 0, 1],[1, 1, 1]]], [[[1, 1, 1],[1, 0, 1],[1, 1, 1]]]], dtype=torch.float)   
    params[0].data = nn.parameter.Parameter(pm)
    params[0].data.requires_grad=False

    pm = torch.tensor([-2.5, -3])
    params[1].data = nn.parameter.Parameter(pm)
    params[1].data.requires_grad=False


def print_board(board, mark="0", void_mark=" "):
    os.system(CLEAR_STR)
    board_print = np.where(board == 1, mark, void_mark)  # "+", " ", "-", "#", "•"
    print(board_print)

class CNN(nn.Module):
    def __init__(
        self, input_channels: int = 1, n_feature: int = 2, padding_mode: str = "circular"
    ) -> None:
        """
        Simple model that uses convolutions

        :param input_channels: number of channels in the board
        :param n_feature: size of the hidden dimensions to use
        :param padding_mode: type of padding, circular for pacman effect on the board
        """
        super().__init__()
        self.n_feature = n_feature
        self.conv = nn.Conv2d(
            in_channels=input_channels, out_channels=n_feature, kernel_size=3, padding=1, padding_mode=padding_mode
        )

    def forward(self, 
                x: torch.Tensor
        ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, channels, w, h]

        :returns: next board with size [batch, channels, w, h]
        """

        k = self.conv(x)
        k1 = k[:,0,...].unsqueeze(1)
        k2 = k[:,1,...].unsqueeze(1)
        p = 0.6 - torch.abs(k2) + torch.einsum("bcij, bcij -> bcij", x, torch.abs(k2) - torch.abs(k1))

        # 0.6 insted of 1 because if x is dead and x has 2 or 4 neighb. => p=0 => sigm(p)=0.5
        # while with 0.6 p < 0
        # 0.6 implies that p>0 for k(x) in (2.4, 3.6)

        #x = torch.heaviside(p, torch.tensor(0.))  # derivative not implemented
        x = torch.sigmoid(p)

        return x

def count_parameters(model: torch.nn.Module) -> int:
  """ Counts the number of trainable parameters of a module
  
  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average

    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager

        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager


def test_model(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str = "cuda",
) -> Dict[str, Union[float, Callable[[Optional[float]], float]]]:
    """Compute model accuracy on the test set

    :param test_dl: the test dataloader
    :param model: the model to train
    :returns: computed accuracy
    """
    model.eval()
    test_loss_averager = make_averager()  # mantain a running average of the loss
    correct = 0
    for data, target in test_loader:

        # send to device
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = torch.einsum("bcij -> b", torch.abs(target-output))
        correct += torch.sum(torch.abs(loss)<1).item()
        loss = torch.sum(loss)
        test_loss_averager(loss)


    return {
        "accuracy": 100.0 * correct / len(test_set),
        "loss_averager": test_loss_averager,
        "correct": correct,
    }

def fit(
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    device: str = "cuda",
) -> float:
    """Train the model and computes metrics on the test_loader at each epoch

    :param epochs: number of epochs
    :param train_dl: the train dataloader
    :param test_dl: the test dataloader
    :param model: the model to train
    :param opt: the optimizer to use to train the model

    :returns: accucary on the test set in the last epoch
    """
    tr_track = []
    te_track = []
    for epoch in trange(epochs, desc="train epoch"):
        model.train()
        train_loss_averager = make_averager()  # mantain a running average of the loss

        # TRAIN
        tqdm_iterator = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"batch [loss: None]",
            leave=False,
        )
        for batch_idx, (data, target) in tqdm_iterator:

            # send to device
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = torch.einsum("bcij -> ", torch.abs(target-output))
            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss_averager(loss.item())



        # TEST
        test_out = test_model(test_loader, model, device)
        

        print(
            f"Epoch: {epoch}\n"
            f"Train set: Average loss: {train_loss_averager(None):.4f}\n"
            f"Test set: Average loss: {test_out['loss_averager'](None):.4f}, "
            f"Accuracy: {test_out['correct']}/{len(test_set)} "
            f"({test_out['accuracy']:.0f}%)\n"
        )

        tr_track.append(train_loss_averager(None))
        te_track.append(test_out['loss_averager'](None))
        
    #models_accuracy = test_out['accuracy']
    #return test_out['accuracy']
    return tr_track, te_track



def get_model_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Encapsulate the creation of the model's optimizer, to ensure that we use the
    same optimizer everywhere

    :param model: the model that contains the parameter to optimize

    :returns: the model's optimizer
    """
    return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # return optim.SGD(model.parameters(), lr=0.01, momentum=0.1, weight_decay=1e-5)




##### MODELS

os.system(CLEAR_STR)

dim = 50
iters = 50
sleep_time = 0.1
manual_mode = False

#pad_mode = "zeros"
pad_mode = "circular"   # pacman effect on the grid



cnn = CNN(padding_mode=pad_mode)

gol = CNN(padding_mode=pad_mode)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Fixing weights for the real GOL
gol_weights(gol)



##### Training


print(f'Using device: {device}') 
cnn.to(device)
#gol.to(device)

# Define the number of the epochs
epochs = 100

train_dim = 5000
test_dim = 100    # TODO: investigate why the GPU memory run out rapidly (a huge spike of memory allocation) if test_dim is like >500
batch_size = 50


train_set = []
for i in range(train_dim):
    inp = torch.randint(low=0, high=2, size=(1,dim,dim), dtype=torch.float)#, device=device)
    with torch.no_grad():
        out = gol(inp.unsqueeze(0))
        out = out.squeeze(0)
    train_set.append((inp,out))

test_set = []
for i in range(test_dim):
    inp = torch.randint(low=0, high=2, size=(1,dim,dim), dtype=torch.float)#, device=device)
    with torch.no_grad():
        out = gol(inp.unsqueeze(0))
        out = out.squeeze(0)
    test_set.append((inp,out))


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    #shuffle=True,
)


optimizer = get_model_optimizer(cnn)

print(f'Number of parameters: {count_parameters(cnn)}')

tr_track, te_track = fit(epochs=epochs, 
    train_loader=train_loader,
    test_loader=test_loader,
    model=cnn,
    opt=optimizer, 
    device=device)

print(list(cnn.parameters()))
#print(list(gol.parameters()))


plt.plot(tr_track, label="train loss")
plt.plot(te_track, label="test loss")
plt.show()

input("\nPress Enter to run simulation")

##### Running

cnn.to("cpu")
cnn.eval()
gol.to("cpu")
#run(cnn, dim, iters, sleep_time)


initial_board = torch.randint(low=0, high=2, size=(1,1,dim,dim), dtype=torch.float)

o1 = run_init(cnn, initial_board, iters, sleep_time)#, manual_mode=True)
input()
o2 = run_init(gol, initial_board, iters, sleep_time)
input()
os.system(CLEAR_STR)
loss = torch.einsum("bcij -> ", torch.abs(o1-o2))
print(f"\n\nSimulated Board Loss: {loss.item()}\n")




## Observations:

# since the rule is the same for test and train
# the learning follow almost the same curve for both
# the errors. The data is so unbiased that it may has 
# no sense to use the test set during the training.


### GOL with Conv2D:  


# k = conv(x, ker), ker = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
# p = x*[1 - |k(x) - 2.5| ] + [1-x]*[ 1 - |k(x) - 3|]
# y = Heaviside(p)

# x alive => x = 1
# p = 1-|k(x)-2.5| > 0 if k(x) == 2 or k(x) == 3
# x dead => 0
# p = 1- |k(x) -3| > 0 if k(x) == 3 

# -2.5 and -3 are biases

# So we can define 2 filters
# f1(x) = k(x) + bias = k(x) -2.5
# f2(x) = k(x) + bias = k(x) -3

# reformuling:

# p = 1 - |k(x) - 3| + x*[ |k(x) - 3|- |k(x) - 2.5| ]
# p = 1 - |f2(x)| + x*[ |f2(x)|- |f1(x)| ]

# Also possible: k1 = [[1, 1, 1], [1, A, 1], [1, 1, 1]]  with bias b = -2.5 - A
# since k1 is effective only when x is alive

