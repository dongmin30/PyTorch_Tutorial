import matplotlib.pyplot as plt
import numpy as np
import torchvision

# 이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_rand_images(dataloader, classes, batch_size):
  # 학습용 이미지를 무작위로 가져오기
  dataiter = iter(dataloader)
  images, labels = next(dataiter)

  # 이미지 보여주기
  imshow(torchvision.utils.make_grid(images))
  # 정답(label) 출력
  print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
