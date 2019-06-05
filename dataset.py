#%%
# Imports
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, ImageOps

#%%
class Cancer(Dataset):
    #augmenting the data will create shifted samples of the images to improve training
    #preprocessing the data will enhace the edges of the data
    def __init__(self, data_type, data_dir, labels_dir, aug=False, preprocess=True, ToTensor=True):
        super(Cancer, self).__init__()
        self.data_dir = data_dir
        self.length = 200
        self.labels_dir = labels_dir

        w_dir = os.path.join(self.data_dir, data_type)

        #files will have the list of filenames
        files = os.listdir(w_dir)
        #only take some of the files
        files = files[:self.length]

        #store in self
        self.file_list = files

        dataset = []
        dataset_dic = {}
        # test_filename = 'test.png'
        NEW_HEIGHT = 500
        NEW_WIDTH = 1000
        for f in tqdm(files, desc=data_type):
            p = os.path.join(w_dir, f)
            if preprocess:
                #read in the image
                image = Image.open(p)
                #convert to grayscale
                gray_image = ImageOps.grayscale(image)
                # gray_image = image
                #equalize the histogram
                gray_equalized = ImageOps.equalize(gray_image)
                #resize the image
                gray_equalized = gray_equalized.resize((NEW_WIDTH, NEW_HEIGHT))
                #could also be a move????
                #look at PIL.ImageEnhance as well
                # gray_auto_contrast = ImageOps(gray_image)
                c = np.array(gray_equalized).reshape((1, NEW_HEIGHT, NEW_WIDTH)).astype(float)
            else:
                image = Image.open(p)
                # image_resize = image.resize((NEW_WIDTH, NEW_HEIGHT))
                # f = np.array(image).reshape((1, NEW_HEIGHT, NEW_WIDTH))
                c = np.array(image)
                image_shape = c.shape

                c = c.reshape((3, image_shape[1], image_shape[1]))
                # print('image shape',c.shape)
                
                # print('f shape:', f.shape)
            #f should be 2D now
            # print(f.shape)
            if ToTensor:
                #convert the ndarray to a tensor
                # print('converting')
                c = self.convert(c)

            #AUGMENT IF NEEDED
            if aug:
                # we can add a rotated or noisey image to the dataset 
                # a certain percentage of the time
                #PIL rotate
                pass
            dataset_dic[f] = c
            dataset.append(c)

        self.dataset = dataset
        self.dataset_dic = dataset_dic
        # print('length of dataset:', len(self.dataset))

        # self.labels, self.labels_one_hot = self.get_labels()
        self.labels = self.get_labels()

        # print('length of labels:',len(self.labels))

        # print('the files:')
        # for i in self.file_list:
        #     print(i)
        # print(self.labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        #return the label as well
        # return self.dataset[ind], self.labels_one_hot[ind]
        item = {
            'data': self.dataset_dic[self.file_list[ind]],
            # 'label': self.labels_one_hot[ind].reshape(1,-1)
            'label': self.labels[self.file_list[ind]],
            'file': self.file_list[ind]
        }
        return item
    
    def convert(self, ndarray):
        tensor = torch.from_numpy(ndarray).double()
        return tensor
    
    def get_labels(self):
        #treat labels as a one-hot vector?
        l_dict = {}
        with open(self.labels_dir) as c:
            l = [i.split(',') for i in c]
            for i, j in l:
                if i == 'id':
                    continue
                l_dict[i+'.tif'] = int(j)
            # l_dict = { i+'.tif':int(j) for i, j in l}
        # del l_dict['id.tif']

        # l_unique = np.unique([_ for _ in l_dict.values()])
        # d = {l_unique[i]: i for i in range(l_unique.shape[0])}
        # labels = [ d[l_dict[i]] for i in self.file_list]
        # length = len(labels)
        # m = max(labels)
        # labels_one_hot = np.zeros((length, m+1))
        # labels_one_hot[np.arange(length), labels] = 1
        # # print('one hot labels:',type(labels_one_hot))
        # # print(labels_one_hot[:10])
        # labels_one_hot = labels_one_hot.astype(int)
        # # print(labels_one_hot[:10])
        
        # return labels_one_hot
        return l_dict



#%%
