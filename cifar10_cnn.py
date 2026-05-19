"""CIFAR-10 Image Classifier — Residual CNN (PyTorch)"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings; warnings.filterwarnings("ignore")

BATCH_SIZE,EPOCHS,LR=128,20,0.001
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
print(f"Device: {DEVICE}")

train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
test_tf=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
train_ds=datasets.CIFAR10("./data",train=True, download=True,transform=train_tf)
test_ds =datasets.CIFAR10("./data",train=False,download=True,transform=test_tf)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True, num_workers=0,pin_memory=True)
test_loader =DataLoader(test_ds, batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)

class ConvBlock(nn.Module):
    def __init__(self,ic,oc,stride=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ic,oc,3,stride=stride,padding=1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(True),
                               nn.Conv2d(oc,oc,3,padding=1,bias=False),nn.BatchNorm2d(oc))
        self.skip=nn.Sequential(nn.Conv2d(ic,oc,1,stride=stride,bias=False),nn.BatchNorm2d(oc)) if stride!=1 or ic!=oc else nn.Identity()
        self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.net(x)+self.skip(x))

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(True))
        self.l1=ConvBlock(32,64); self.l2=ConvBlock(64,128,2); self.l3=ConvBlock(128,256,2); self.l4=ConvBlock(256,512,2)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.4),nn.Linear(512,10))
    def forward(self,x): return self.head(self.l4(self.l3(self.l2(self.l1(self.stem(x))))))

model=CIFAR10Net().to(DEVICE)
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

def evaluate(loader):
    model.eval(); c=t=ls=0; pa=[]; la=[]
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE); out=model(imgs)
            ls+=criterion(out,labels).item()*imgs.size(0); p=out.argmax(1)
            c+=(p==labels).sum().item(); t+=labels.size(0)
            pa.extend(p.cpu().numpy()); la.extend(labels.cpu().numpy())
    return ls/t,c/t,np.array(pa),np.array(la)

history={"tl":[],"vl":[],"ta":[],"va":[]}; best_acc=0; best_state=None
for epoch in range(1,EPOCHS+1):
    model.train(); tl=tc=tt=0
    for imgs,labels in train_loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE); optimizer.zero_grad()
        out=model(imgs); loss=criterion(out,labels); loss.backward(); optimizer.step()
        tl+=loss.item()*imgs.size(0); tc+=(out.argmax(1)==labels).sum().item(); tt+=labels.size(0)
    scheduler.step(); tl/=tt; ta=tc/tt; vl,va,_,_=evaluate(test_loader)
    history["tl"].append(tl); history["vl"].append(vl); history["ta"].append(ta); history["va"].append(va)
    print(f"Epoch {epoch}: train={ta:.4f} val={va:.4f}")
    if va>best_acc: best_acc=va; best_state={k:v.clone() for k,v in model.state_dict().items()}

model.load_state_dict(best_state)
_,acc,preds,labels=evaluate(test_loader)
print(f"\nBest Acc: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(labels,preds,target_names=CLASSES))
cm=confusion_matrix(labels,preds)
plt.figure(figsize=(10,8)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=CLASSES,yticklabels=CLASSES)
plt.title(f"Confusion Matrix (Acc={acc:.4f})"); plt.tight_layout()
plt.savefig("confusion_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
torch.save(model.state_dict(),"cifar10_cnn.pth")
print("\n✅ Done!")
