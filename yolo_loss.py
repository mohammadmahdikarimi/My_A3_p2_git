import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt

class YoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj, use_gpu):
        super(YoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.C = 20  #number of bounding classes
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.use_gpu = use_gpu

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    
    def get_class_prediction_loss(self, classes_pred, classes_target):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)

        Returns:
        class_loss : scalar
        """

        return torch.sum((classes_pred - classes_target)**2) 
        #return class_loss
    
    
    def get_regression_loss(self, box_pred_response, box_target_response):   
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 5)
        box_target_response : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar
        
        """
        #box_pred_response = box_pred[coord_response_mask].view(-1, 5)
        #box_target_response = box_target[coord_response_mask].view(-1, 5)
        reg_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction = 'sum') +\
                   F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction = 'sum')
        
        
        return reg_loss 
    
    def get_contain_conf_loss(self, box_pred_response, box_target_response_iou):
        """
        Parameters:
        box_pred_response : (tensor) size ( -1 , 5)
        box_target_response_iou : (tensor) size ( -1 , 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        contain_loss : scalar
        
        """
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction = 'sum')
        
        return contain_loss
    
    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):
        """
        Parameters:
        target_tensor : (tensor) size (batch_size, S , S, 30)
        pred_tensor : (tensor) size (batch_size, S , S, 30)
        no_object_mask : (tensor) size (batch_size, S , S, 30)

        Returns:
        no_object_loss : scalar

        Hints:
        1) Create a 2 tensors no_object_prediction and no_object_target which only have the 
        values which have no object. 
        2) Have another tensor no_object_prediction_mask of the same size such that 
        mask with respect to both confidences of bounding boxes set to 1. 
        3) Create 2 tensors which are extracted from no_object_prediction and no_object_target using
        the mask created above to find the loss. 
        """

        n_elements = self.B * 5 + self.C
        noobj_target = target_tensor[no_object_mask.bool()].view(-1,n_elements)
        noobj_pred = pred_tensor[no_object_mask.bool()].view(-1,n_elements)

        if self.use_gpu:
            noobj_target_mask = torch.cuda.ByteTensor(noobj_target.size())
        else:
            noobj_target_mask = torch.ByteTensor(noobj_target.size())
        noobj_target_mask.zero_()
        for i in range(self.B):  #confidences of bounding boxes set to 1. 
            noobj_target_mask[:,i*5+4] = 1
        noobj_target_c = noobj_target[noobj_target_mask.bool()] # only compute loss of c size [2*B*noobj_target.size(0)]
        noobj_pred_c = noobj_pred[noobj_target_mask.bool()]
        no_object_loss = F.mse_loss(noobj_pred_c, noobj_target_c, reduction='sum')

        return no_object_loss
        
    
    
    def find_best_iou_boxes(self, box_target, box_pred):
        """
        Parameters: 
        box_target : (tensor)  size (-1, 5)
        box_pred : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns: 
        box_target_iou: (tensor)
        contains_object_response_mask : (tensor)

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) Set the corresponding contains_object_response_mask of the bounding box with the max iou
        of the 2 bounding boxes of each grid cell to 1.
        3) For finding iou's use the compute_iou function
        4) Before using comtpute preprocess the bounding box coordinates in such a way that 
        if for a Box b the coordinates are represented by [x, y, w, h] then 
        x, y = x/S - 0.5*w, y/S - 0.5*h ; w, h = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height. 
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        5) Set the confidence of the box_target_iou of the bounding box to the maximum iou
        
        """
        if self.use_gpu:
            coo_response_mask = torch.cuda.ByteTensor(box_target.size())
            box_target_iou = torch.cuda.FloatTensor(box_target.size())
        else:
            coo_response_mask = torch.ByteTensor(box_target.size())
            box_target_iou = torch.FloatTensor(box_target.size())
        coo_response_mask.zero_()
        box_target_iou.zero_()
        for i in range(0,box_target.size()[0],self.B):
            box_p = box_pred[i:i+self.B]
            box_t = box_target[i:i+self.B]

            # compute [x1,y1,x2,y2] w.r.t. top left and bottom right coordinates separately
            b1x1y1 = box_p[:,:2]/self.S -box_p[:,2:4]/2 # [N, (x1,y1)=2]
            b1x2y2 = box_p[:,:2]/self.S +box_p[:,2:4]/2 # [N, (x2,y2)=2]
            b2x1y1 = box_t[:,:2]/self.S -box_t[:,2:4]/2 # [M, (x1,y1)=2]
            b2x2y2 = box_t[:,:2]/self.S +box_t[:,2:4]/2 # [M, (x1,y1)=2]
            box_p = torch.cat((b1x1y1.view(-1,2), b1x2y2.view(-1, 2)), dim=1) # [N,4], 4=[x1,y1,x2,y2]
            box_t = torch.cat((b2x1y1.view(-1,2), b2x2y2.view(-1, 2)), dim=1) # [M,4], 4=[x1,y1,x2,y2]
            iou = self.compute_iou(box_p, box_t)

            
            max_iou, max_index = iou.max(0)
            if self.use_gpu:
                max_index = max_index.data.cuda()
            else:
                max_index = max_index.data
            coo_response_mask[i+max_index] = 1
            box_target_iou[i+max_index[0]][-1:] = max_iou[0]


            

        return box_target_iou, coo_response_mask
        
    
    
    
    def forward(self, pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30)
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes
        
        target_tensor: (tensor) size(batchsize,S,S,30)
        
        Returns:
        Total Loss
        '''
        n_elements = self.B * 5 + self.C
        batch = target_tensor.size(0)
        #target_tensor = target_tensor.view(batch,-1,n_elements)
        #pred_tensor = pred_tensor.view(batch,-1,n_elements)

        N = pred_tensor.size()[0]
        
        total_loss = None
        
        # Create 2 tensors contains_object_mask and no_object_mask 
        # of size (Batch_size, S, S) such that each value corresponds to if the confidence of having 
        # an object > 0 in the target tensor.
        
        contains_object_mask = target_tensor[:,:,:,4] > 0
        contains_no_object_mask = target_tensor[:,:,:,4] == 0

        # Create a tensor contains_object_pred that corresponds to 
        # to all the predictions which seem to confidence > 0 for having an object
        
        # Split this tensor into 2 tensors :
        # 1) bounding_box_pred : Contains all the Bounding box predictions of all grid cells of all images
        # 2) classes_pred : Contains all the class predictions for each grid cell of each image
        # Hint : Use contains_object_mask
        contains_object_mask = contains_object_mask.unsqueeze(-1).expand_as(target_tensor)
        contains_no_object_mask = contains_no_object_mask.unsqueeze(-1).expand_as(target_tensor)


        coord_pred = pred_tensor[contains_object_mask].view(-1,n_elements)
        class_pred = coord_pred[:,self.B*5:]
        box_pred = coord_pred[:,:self.B*5].contiguous().view(-1,5)

        # Similarly as above create 2 tensors bounding_box_target and
        # classes_target.
        
        coord_target = target_tensor[contains_object_mask].view(-1,n_elements)
        class_target = coord_target[:,self.B*5:]
        box_target = coord_target[:,:self.B*5].contiguous().view(-1,5)

        noobj_target = target_tensor[contains_no_object_mask].view(-1,n_elements)
        noobj_pred = pred_tensor[contains_no_object_mask].view(-1,n_elements)

        # Compute the No object loss here
        noobj_loss = self.get_no_object_loss(target_tensor, pred_tensor, contains_no_object_mask.view(pred_tensor.size()))
        
        

        # Compute the iou's of all bounding boxes and the mask for which bounding box 
        # of 2 has the maximum iou the bounding boxes for each grid cell of each image.
        
        box_target_iou, contains_object_response_mask = self.find_best_iou_boxes(box_target, box_pred)
        
        # Create 3 tensors :
        # 1) box_prediction_response - bounding box predictions for each grid cell which has the maximum iou
        # 2) box_target_response_iou - bounding box target ious for each grid cell which has the maximum iou
        # 3) box_target_response -  bounding box targets for each grid cell which has the maximum iou
        # Hint : Use contains_object_response_mask

        box_prediction_response = box_pred[contains_object_response_mask.bool()].view(-1, 5)
        box_target_response = box_target[contains_object_response_mask.bool()].view(-1, 5)
        box_target_response_iou = box_target_iou[contains_object_response_mask.bool()].view(-1, 5).detach()
        
        # Find the class_loss, containing object loss and regression loss
        
        noobj_loss = self.get_no_object_loss(target_tensor, pred_tensor, contains_no_object_mask)
        contain_loss = self.get_contain_conf_loss(box_prediction_response, box_target_response_iou)
        reg_loss = self.get_regression_loss(box_prediction_response, box_target_response)
        class_loss = self.get_class_prediction_loss(class_pred, class_target)

        
        

        total_loss = (self.l_coord * reg_loss + contain_loss + self.l_noobj * noobj_loss + class_loss)/batch
        return total_loss




