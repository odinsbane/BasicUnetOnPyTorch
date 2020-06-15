#!/usr/bin/env python3

import torch.nn
import torch.autograd
import torch.nn.functional
import skimage.io
import numpy
import click
import json, time

import torch.utils.data

class IndexingImagePair(torch.utils.data.IterableDataset):
    def __init__(self, image, labels, input_shape, stride=None, output_shape = None, encoder = None):
        if stride is None:
            stride = [i//2 for i in input_shape]
        self.output_channels = labels.shape[0]
        
        self.input_shape = input_shape
        self.stride = stride
        self.image = image
        self.labels = labels
        self.indexes = None
        
        print("shapes", self.input_shape, self.image.shape)
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        r = self.indexes[index]
        return numpy.array( self.image[
                r[0]:r[0] + self.input_shape[0], 
                r[1]:r[1] + self.input_shape[1],
                r[2]:r[2] + self.input_shape[2],
                r[3]:r[3] + self.input_shape[3]
            ] , dtype='float' ), numpy.array(self.labels[
                r[0]:r[0] + self.output_channels, 
                r[1]:r[1] + self.input_shape[1],
                r[2]:r[2] + self.input_shape[2],
                r[3]:r[3] + self.input_shape[3]
            ], dtype='float' )
             
    def __iter__(self):
        if self.indexes is None:
            generateIndexes()
        return iter( self[i] for i in range(len(self.indexes)) )
    
    def len(self):
        return len(self.indexes)
    
    def generateIndexes(self):
        self.indexes = []
        
        c0 = 0 #always use all of the channels.
        for z0 in range(0, self.image.shape[1] - self.input_shape[1], self.stride[1]):
            for y0 in range(0, self.image.shape[2] - self.input_shape[2], self.stride[2]):
                for x0 in range(0, self.image.shape[3] - self.input_shape[3], self.stride[3]):
                    self.indexes.append((c0, z0, y0, x0))
    
    
    
class DeviceAdapter:
    def getTensor(self, data):
        return None
    def numpyData(self, tensor):
        return tensor.cpu().detach().numpy()
    def prepareModel(self, model):
        return None

class CpuAdapter(DeviceAdapter):
    def __init__(self):
        self.device = torch.device('cpu')
        
    def getTensor(self, data):
        return torch.Tensor(numpy.array(data, dtype='float32'))
    def prepareModel(self, model):
        model.to("cpu:0")

class CudaAdapter(DeviceAdapter):
    def __init__(self, id=0):
        self.device = torch.device("cuda:%d"%id)
    def getTensor(self, data):
        return torch.tensor(numpy.array(data, dtype='float32'), device = self.device)
    def prepareModel(self, model):
        model.to(self.device)

class UNet(torch.nn.Module):
    def __init__(self, depth, filters, input_shape=None):
        super(UNet, self).__init__()
        self.contracting = []
        self.expanding = []
        self.depth = depth
        self.filters = filters
        if input_shape is None:
            input_shape = (1,)
        i_c = input_shape[0]
        o_c = filters
        kernel_size=(3,3,3)
        pooling = (1,2,2)
        self.pooling = torch.nn.MaxPool3d(pooling)
        padding = tuple(k//2 for k in kernel_size)
        self.pad =torch.nn.ReplicationPad3d((0,1,0,1,0,0))
        self.inputShape = input_shape
        for i in range(depth):
            d= []
            d.append(torch.nn.Conv3d(i_c, o_c, kernel_size, padding=padding))
            i_c = o_c
            o_c = 2*o_c
            d.append(torch.nn.Conv3d(i_c, o_c, kernel_size, padding=padding))
            i_c = o_c
            self.contracting.append(d)
        up_padding = padding
        for i in range(depth-1):
            d = []
            d.append(torch.nn.ConvTranspose3d(o_c, o_c, kernel_size, stride = pooling, padding=padding))
            n_c = o_c//2
            d.append(torch.nn.Conv3d(o_c + n_c, n_c, kernel_size, padding = padding), )
            d.append(torch.nn.Conv3d(n_c, n_c, kernel_size, padding = padding))
            o_c = n_c
            self.expanding.append(d)
        if o_c != 2*filters:
            print("borked!", o_c, n_c, 2*filters)
            
        self.final = torch.nn.Conv3d(2*filters, 1, kernel_size, padding = padding)
        self.affix_layers()
    def forward(self, x):
        cross = []
        t = x
        for depth, con in enumerate(self.contracting[:-1]):
            #print("d:", depth)
            for i in con:
                t = torch.nn.functional.relu(i(t))
                #print("  t: ", t.shape)
            cross.append(t)
            t = self.pooling(t)
        cross.reverse()

        for i in self.contracting[-1]:
            t = torch.nn.functional.relu(i(t))

        for up, exp in enumerate(self.expanding):
            #print("before tconv: ", t.shape)
            t = torch.nn.functional.relu(exp[0](t))
            #print("after tconv: ", t.shape)
            t = self.pad(t)
            t = torch.cat([cross[up], t], dim=1)
            #print("catted: ", t.shape)
            for i in exp[1:]:
                t = torch.nn.functional.relu(i(t))
                #print("  c: ", t.shape)

        return torch.sigmoid(self.final(t))
    def affix_layers(self):
        """
            Attach the layers as first level members of the model for serialization and
            loading.
        """
        for i, row in enumerate(self.contracting):
            for j, e in enumerate(row):
                self.__setattr__("conv3d-%s-%s"%(i,j), e)
        for i, row in enumerate(self.expanding):
            for j, e in enumerate(row):
                self.__setattr__("conv3d-%s-%s"%(i+len(self.contracting),j), e)
                
    
    def shift_device(self, device):
        for i in self.contracting:
            for ii in i:
                ii.to(device)
        for j in self.expanding:
            for jj in j:
                jj.to(device)
        self.to(device)
        
    def setInputShape(self, shp):
        self.inputShape=shp

class IncorrectPixelLoss(torch.nn.modules.loss._Loss):
    def __init__(self, sigma=0.1):
        super(IncorrectPixelLoss, self).__init__()
        self.sigma = sigma
        
    def forward(self, predicted, expected):
        exp = torch.clamp(expected, 0, 1);
        nexp = (1 - exp)

        fn = (exp)*(1 - predicted)
        fp = (nexp)*predicted
        
        return ( torch.sum(fp) + torch.sum(fn) + self.sigma ) / ( torch.sum( exp ) + self.sigma )

class DiceLoss(torch.nn.modules.loss._Loss):
    def __init__(self, sigma=0.1):
        super(DiceLoss, self).__init__()
        self.sigma = sigma
    def forward(self, predicted, expected):
        dot = predicted*expected
        return - 2 * ( torch.sum(dot) + self.sigma ) / (torch.sum( predicted ) + torch.sum( expected ) + self.sigma )

    
def createUnet(depth=3, filters=32, input_shape=None):
    return UNet(depth, filters, input_shape)

def saveModel(model, path, optimizer = None, epoch=-1):
    checkpoint = {
        "depth":model.depth,
        "filters":model.filters,
        "input_shape":model.inputShape,
        "model_state_dict":model.state_dict(),
        "epoch": epoch
     }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"]=optimizer.state_dict()
    torch.save(checkpoint, path)

def loadModel(path, device_adapter):
    values = torch.load(path, map_location=device_adapter.device)
    depth = values["depth"]
    filters = values["filters"]
    model = UNet(depth, filters, values["input_shape"]) 
    try:
        model.load_state_dict(values["model_state_dict"])
    except:
        for item in values["model_state_dict"]:
            print(item, values[item])
            return -1
    if "input_shape" in values:
        pass
    #model.input_shape=values["input_shape"]
    optimizer = None
    
    if "optimizer_state_dict" in values:
        pass
    return model, optimizer



def trainModel(model, optimizer=None, datasets=[], batch_size=32, device_adapter = None ):
    
    loaders = []
    for dataset in datasets:
        dataset.generateIndexes()
        loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    print(len(loaders), " loaders.")
    #sets train mode
    model.train()

    #loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_fn = DiceLoss()
    #loss_fn = IncorrectPixelLoss()
    
        
    device_adapter.prepareModel(model)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        
    for t in range(500):
        #permutation = torch.randperm( x.size()[0] )
        epoch_loss = 0;
        st = time.time()
        
        counter = 0
        for loader in loaders:
            for i, (x, y) in enumerate(loader):
                y_pred = model( device_adapter.getTensor(x))
                optimizer.zero_grad()
                loss = loss_fn(y_pred, device_adapter.getTensor(y))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                counter += 1
        saveModel(model, "testing-diceloss.pth")
        print("epoch", t, time.time() - st, epoch_loss/counter)

def getPredictionChunks(rs, image, model, device_adapter):
    isp = numpy.array(model.inputShape)
    chunks = []
    for r0 in rs:
        
        chunks.append(image[
                                r0[0] : r0[0] + isp[0],
                                r0[1] : r0[1] + isp[1],
                                r0[2] : r0[2] + isp[2],
                                r0[3] : r0[3] + isp[3]
                                ])
    ten = device_adapter.getTensor(numpy.array(chunks, dtype="float32"))
    prediction = model(ten)
    
    return prediction

def forStrideRange(mx, s, p):
    for i in range(0, mx, s):
        if i + p > mx:
            yield mx - p
        else:
            yield i
    


def predictImage(model, img, device_adapter):
    input_shape = numpy.array(model.inputShape)
    batch_size = 64
    stride = input_shape//2
    rs = []
    output = numpy.zeros(img.shape, dtype="uint16")
    c0 = 0
    for x0 in forStrideRange(img.shape[-1],stride[-1], input_shape[-1]):
        for y0 in forStrideRange(img.shape[-2], stride[-2], input_shape[-2]):
            for z0 in forStrideRange(img.shape[-3], stride[-3], input_shape[-3]):
                rs.append(numpy.array([c0, z0, y0, x0]))
                if len(rs)==batch_size:
                    pred = getPredictionChunks(rs, img, model, device_adapter)
                    pred = device_adapter.numpyData(pred)
                    
                    for r, chunk in zip(rs, pred):
                        output[r[0]:r[0] + input_shape[0], 
                               r[1]:r[1] + input_shape[1], 
                               r[2]:r[2] + input_shape[2],
                               r[3]:r[3] + input_shape[3]
                               ] = (chunk[0]>0.5)*255
                    rs = []
    return output
import sys

@click.command()
@click.argument("action")
@click.argument("name")
@click.option("-d", "devices", envvar="CUDA_VISIBLE_DEVICES")
def createModelAction(action, name, devices):
    model = createUnet(3, 32, (1, 5, 64, 64))
    saveModel(model, name)


img = "original.tif"
seg = "skeleton.tif"


training_images = [ (img, seg)]

@click.command()
@click.argument("action")
@click.argument("model")
@click.argument("out", default=None, required=False)
@click.option("-d", "devices", envvar="CUDA_VISIBLE_DEVICES")
def trainModelAction(action, model, out, devices):
    device_adapter = getDeviceAdapter(devices);
    
    model, opt = loadModel(model, device_adapter)
    #TODO fix
    image_pairs = []
    for img, seg in training_images:
        xs = skimage.io.imread(img)
        xs = numpy.array([xs])
        
        ys = skimage.io.imread(seg)
        #ys = ys.reshape( ( 1, *ys.shape) )
        ys = numpy.array([ys])
        print(xs.shape, ys.shape)
        image_pairs.append(IndexingImagePair(xs, ys, model.inputShape))
    
    trainModel(model, opt, image_pairs, batch_size=32, device_adapter =  device_adapter)

def getDeviceAdapter(visible_device_string):
    if len(visible_device_string) == 0:
        return CpuAdapter()
    else:
        #find number of devices and return 
        return CudaAdapter(0)
    

@click.command()
@click.argument("action")
@click.argument("model", type=click.Path(exists=True) )
@click.argument("image", type=click.Path(exists=True))
@click.argument("prediction")
@click.option("-d", "devices", envvar="CUDA_VISIBLE_DEVICES")
def predictImageAction(action, model, image, prediction, devices):
    device_adapter = getDeviceAdapter(devices)
    model, optimizer = loadModel(model, device_adapter)
    device_adapter.prepareModel(model)
    
    img = skimage.io.imread(image)
    if len(img.shape)==3:
        img = numpy.reshape(img, (1, *img.shape))
    pred = predictImage(model, img, device_adapter)
    skimage.io.imsave(prediction, pred)



if __name__=="__main__":

    actions = {
     'c':createModelAction,
     't':trainModelAction,
     'p':predictImageAction
    }
    
    actions[sys.argv[1]]()
