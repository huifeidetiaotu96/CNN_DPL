import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


transform = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 自定义数据集，只返回数据，不返回标签 仅限训练集
class DiabetesDataset(Dataset):
    def __init__(self, root=None):
        imgs = os.listdir(root)
        # self.label = int(root.spilt('/')[-1])
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert('RGB')
        data = self.transforms(pil_img)
        # label = self.label
        return data

    def __len__(self):
        return len(self.imgs)
