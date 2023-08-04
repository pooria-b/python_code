device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        
        
        
def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=None, num_classes):
  model.train()
  loss_train = AverageMeter()
  acc_train = Accuracy(task='multiclass',
                                    num_classes=num_classes).to(device)
  with tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      if epoch is not None:
        tepoch.set_description(f'Epoch {epoch}')
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      loss = loss_fn(outputs, targets)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item())
      acc_train(outputs, targets.int())

      tepoch.set_postfix(loss=loss_train.avg,
                         accuracy=100.*acc_train.compute().item())
  return model, loss_train.avg, acc_train.compute().item()


def evaluate(model, valid_loader, loss_fn, num_classes):
  model.eval()
  with torch.no_grad():
    loss_valid = AverageMeter()
    acc_valid = Accuracy(task='multiclass',
                                     num_classes=num_classes).to(device)
    for i, (inputs, targets) in enumerate(valid_loader):
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      loss = loss_fn(outputs, targets)

      loss_valid.update(loss.item())
      acc_valid(outputs, targets.int())

  return loss_valid.avg, acc_valid.compute()



def diagnose_mv(df, mv_column):
    cols = list(df.columns)
    cols.remove(mv_column)
    flags = df[mv_column].isna()
    fig, ax = plt.subplots(len(cols), 3, 
                           figsize=(len(cols)+3, len(cols)+3), 
                           constrained_layout=True)
    plt.rcParams['axes.grid'] = True
    for i, col in enumerate(cols):
        n1, bins, _ = ax[i, 0].hist(df[col])
        ax[i, 0].set_title(f'{col} with MV')
        #
        n2, _, _ = ax[i, 1].hist(df[col][~flags], bins=bins)
        ax[i, 1].set_title(f'{col} without MV')
        #
        ax[i, 2].bar(bins[:-1], np.abs(n2-n1), width=np.abs(bins[1]-bins[0]))
        ax[i, 2].set_title(f'Difference')