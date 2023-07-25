# 이미지 분류기 학습하기
import os
import torch
import torchvision
import torchvision.transforms as transforms
# 학습용 이미지 랜덤 출력
from cifat10_image import show_rand_images
# 신경망 정의 라이브러리
import torch.nn as nn
import torch.nn.functional as F
# optim을 통해 매개변수 갱신 함수 호출
import torch.optim as optim

# 1. CIFAR10을 불러오고 정규화 하기
transform = transforms.Compose(
  [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 모델 파일 저장 위치
PATH = './cifar_net.pth'

# 학습용 이미지 확인해보기
# show_rand_images(dataloader=trainloader, classes=classes, batch_size=batch_size)

# 2. 합성곱 신경망(Convolution Neural Network - CNN) 정의하기
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  
  # forward만 정의하고 나면 역전파인 backward는 autograd를 사용하서 자동 생성
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
net = Net()

# 3. 손실 함수와 Optimizer 정의하기
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. 신경망 학습하기
if os.path.isfile(PATH) != True :
  for epoch in range(2): # 데이터셋을 수차례 반복 제공
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # [inputs, labels]의 목록인 data로부터 입력을 받은 후
      inputs, labels = data
      
      # 변화도(Gredient) 매개변수를 0으로 만들고
      optimizer.zero_grad()
      
      # 순전파 + 역전파 + 최적화를 한 후
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      
      # 통계를 출력합니다.
      running_loss += loss.item()
      if i % 2000 == 1999: # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0
        
  print('Finished Training')
  
  # 학습 모델 저장
  torch.save(net.state_dict(), PATH)

# 5. 시험용 데이터로 신경망 검사하기
dataiter = iter(testloader)
images, labels = next(dataiter)
# 시험용 데이터 이미지 확인
# show_rand_images(dataloader=testloader, classes=classes, batch_size=batch_size)

# 저장했던 모델 불러오기
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# 전체 데이터셋 동작 체크
correct = 0
total = 0

# 학습 중이 아니므로, 출력에 대한 변화도 계산이 필요없음
with torch.no_grad():
  for data in testloader:
    images, labels = data
    # 신경망에 이미지를 통과시켜 출력을 계산합니다
    outputs = net(images)
    # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# 각 분류(class)에 대한 예측값 계산을 위해 준비
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# 변화도는 여전히 필요하지 않음
with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predictions = torch.max(outputs, 1)
    # 각 분류별로 올바른 예측 수를 모읍니다.
    for label, prediction in zip(labels, predictions):
      if label == prediction:
        correct_pred[classes[label]] += 1
      total_pred[classes[label]] += 1

# 각 분류별 정확도(accuracy)를 출력합니다.
for classname, correct_count in correct_pred.items():
  accuracy = 100 * float(correct_count) / total_pred[classname]
  print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
  
# GPU에서 학습하기 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# CUDA 기기가 존재한다면, 아래 코드가 CUDA 장치를 출력
print(device)

# 이제 해당 메소드를 통해 모든 모듈의 매개변수와 버퍼를 CUDA tensor로 변경
net.to(device)

# 또한, 각 단계에서 입력(input)과 정답(target)도 GPU로 보냄
input, labels = data[0].to(device), data[1].to(device)