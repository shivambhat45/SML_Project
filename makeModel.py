from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
import torch

class Model(nn.Module):
    def __init__(self,layers):
        super(Model, self).__init__()
        self.features = layers[0]
        self.classifier = layers[1]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, args, logger ):
    layersC = []
    layersF = []
    in_channels = cfg[0][-1]

    for i, v in enumerate(cfg):
        # if(len(v) == 4):
        #     non_linearity_activation =  nn.ReLU(inplace=True)
        # else :
        non_linearity_activation =  nn.Tanh()
        if v[0] == 'M':
            layersC += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        elif v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                             logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                             wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                             subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                             name = 'Conv'+str(i)+'_' )
          
            layersC += [conv2d]
            in_channels = out_channels
        elif(v[0] == 'fc'):
            linear = QLinear(in_features=v[1], out_features=v[2],logger=logger,
                             wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                             wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                             subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                             name='FC'+str(i)+'_')
            if i < len(cfg)-1:
                    non_linearity_activation =  nn.ReLU()
                    layersF += [linear, non_linearity_activation]
            else:
                    layersF += [linear]            
    return nn.Sequential(*layersC) , nn.Sequential(*layersF)


def Create_Model( args, logger, arch , pretrained=None):
    model = Model(make_layers(arch, args,logger))
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

