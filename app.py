import torch 
import torchvision 
import numpy as np 
from torch import nn 
from torch.autograd import Variable
from parse_arguments import parse_opts 
from models import mobilenetv2_3d
from spatial_transforms import *
from temporal_transforms import * 
from target_transforms import ClassLabel, VideoID 
from target_transforms import Compose as TargetCompose 
from mean import get_mean, get_std
import test 
from PIL import Image
import cv2 as cv


if __name__ == '__main__':
    opt = parse_opts()  
    opt.resume_path = '/home/ismayil/gesture/results/jester_mobilenetv2_RGB_8_best.pth'
    opt.dataset = 'jester' 
    opt.modality = 'RGB' 
    opt.test = False
    opt.no_train = True 
    opt.no_val = False
    opt.n_classes = 27 
    opt.modality = 'RGB'
    opt.model = 'mobilenetv2'
    opt.sample_duration = 8
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.no_cuda = True
    opt.no_mean_norm = False
    opt.std_norm = True

    device = torch.device("cpu")
    torch.manual_seed(opt.manual_seed)
    # load model 
    model = mobilenetv2_3d.get_model(num_classes=opt.n_classes,sample_size=opt.sample_size,width_mult=opt.width_mult)   
    model = model.to(device)    
    checkpoint = torch.load(opt.pretrained_path, map_location=device)
    pretrained_state_dict = checkpoint['state_dict']
    pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if 'module.features.' not in k}
    pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if 'module.lstm_layers.' not in k}
    pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if 'module.classifier.' not in k}
    model_dict = model.state_dict()
    model_dict.update(pretrained_state_dict)
    model.load_state_dict(model_dict)    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params) 
    
    
    
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value),
        norm_method
    ])
    temporal_transform = TemporalCenterCrop(opt.sample_duration)
    target_transform = ClassLabel() 

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    array_sequence = []
    count = 0 
    
    #torch_sequence = torch.zeros((1,3,8,112,112),device = device).cpu()
    while True:
        # set the model to evaluation mode
        count = count + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
       
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            # Display the resulting frame
        # Model part
        n_frames = 8            
        
        target_transform = ClassLabel()

       
        if spatial_transform is not None:
            spatial_transform.randomize_parameters()
        frame = spatial_transform(Image.fromarray(frame))  
        
        array_sequence.append(frame)
        if len(array_sequence) ==  n_frames:                                
            torch_sequence = torch.stack(array_sequence)
            torch_sequence = torch_sequence.unsqueeze_(0)              
            torch_sequence = torch_sequence.permute(0,2,1,3,4) 
            torch_sequence = torch_sequence.to(device)

            with torch.no_grad():
                model.eval() 
                inputs = Variable(torch_sequence)                
                output = model(torch_sequence)
                out = torch.nn.functional.softmax(output, dim=1)
                pred = np.argmax(out.detach().numpy())
            array_sequence = []
                
            count = 0
            print(out)
               
        
        

        if cv.waitKey(3) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()




