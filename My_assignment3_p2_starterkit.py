
import torch
from yolo_loss import YoloLoss

# some help function
def test_error(diff, test='', eps=1e-5):
    if isinstance(diff, torch.Tensor):
        diff = diff.cpu().detach().float()
    print('Error is %f.' % diff)
    if diff < eps:
        print("- You pass the test for %s!" % test)
    else:
        print("- emm.. something wrong. maybe double check your implemention.")

#====================


# don't change the hyperparameter here
yolo = YoloLoss(S=14, B=2, l_coord=5, l_noobj=0.5)


#====================

# load test cases
func_name = 'get_class_prediction'
input_data = torch.load("test_cases/%s_input.pt" % func_name)
class_pred = input_data['classes_pred']
class_target = input_data['classes_target']
output_data = torch.load("test_cases/%s_output.pt" % func_name)
gt_loss = output_data['class_loss']

# calculate my implemented loss
my_loss = yolo.get_class_prediction_loss(class_pred, class_target)

print('class_loss_gt: ', my_loss)

# test the difference between my loss and the gt loss
loss_diff = torch.sum((gt_loss - my_loss) ** 2)
test_error(loss_diff, test=func_name)

#====================

# load test cases
func_name = "get_regression"
input_data = torch.load("test_cases/%s_input.pt" % func_name)
box_pred_response = input_data['box_pred_response']
box_target_response = input_data['box_target_response']
output_data = torch.load("test_cases/%s_output.pt" % func_name)
gt_loss = output_data['reg_loss']

# calculate my implemented loss
#my_loss = yolo.get_regression_loss(box_pred_response.cuda(), box_target_response.cuda())
my_loss = yolo.get_regression_loss(box_pred_response, box_target_response)

print('reg_loss_gt: ', my_loss)

# test the difference between my loss and the gt loss
loss_diff = torch.sum((gt_loss - my_loss) ** 2)

test_error(loss_diff, test=func_name)


#====================

func_name = "get_contain_conf"
input_data = torch.load("test_cases/%s_input.pt" % func_name)
box_pred_response = input_data['box_pred_response']
box_target_response_iou = input_data['box_target_response_iou']
output_data = torch.load("test_cases/%s_output.pt" % func_name)
gt_loss = output_data['contain_loss']

# calculate my implemented loss
my_loss = yolo.get_contain_conf_loss(box_pred_response, box_target_response_iou)
print('contain_loss_gt: ', my_loss)


# test the difference between my loss and the gt loss
loss_diff = torch.sum((gt_loss - my_loss) ** 2)

test_error(loss_diff, test=func_name)


#====================

# load test cases input
func_name = "no_object_loss"
input_data = torch.load("test_cases/%s_input.pt" % func_name)
target_tensor = input_data['target_tensor']
pred_tensor = input_data['pred_tensor']
no_object_mask = input_data['no_object_mask']
output_data = torch.load("test_cases/%s_output.pt" % func_name)
gt_loss = output_data['no_object_loss']

# calculate my implemented loss
my_loss = yolo.get_no_object_loss(target_tensor, pred_tensor, no_object_mask)
print('noobj_loss_gt: ', my_loss)


# test the difference between my loss and the gt loss
loss_diff = torch.sum((gt_loss - my_loss) ** 2)

test_error(loss_diff, test=func_name)


#====================


# load test cases input
func_name = "best_iou_boxes"
input_data = torch.load("test_cases/%s_input.pt" % func_name)
bounding_box_target = input_data['bounding_box_target']
bounding_box_pred = input_data['bounding_box_pred']
output_data = torch.load("test_cases/%s_output.pt" % func_name)
gt_box_target_iou = output_data['box_target_iou']
gt_contains_object_response_mask = output_data['contains_object_response_mask']
bounding_box_pred.requires_grad = True
# calculate my implemented loss
my_box_target_iou, my_contains_object_response_mask = yolo.find_best_iou_boxes(bounding_box_target, bounding_box_pred)

# test the error for the first output
iou_diff = torch.sum((gt_box_target_iou - my_box_target_iou) ** 2)
test_error(iou_diff, test="the first output of %s" % func_name) 

# test the error for the second output
mask_diff = torch.sum((gt_contains_object_response_mask.long() - my_contains_object_response_mask.long()) ** 2)
test_error(mask_diff, test="the second output of %s" % func_name) 
print(my_box_target_iou.requires_grad)
print(my_contains_object_response_mask.requires_grad)


#====================


input_data = torch.load("test_cases/full_input.pt")
pred_tensor = input_data['pred_tensor']
target_tensor = input_data['target_tensor']
output_data = torch.load("test_cases/full_output.pt")
gt_loss = output_data['total_loss']

# calculate my implemented loss
my_loss = yolo(pred_tensor, target_tensor)

# test the difference between my loss and the gt loss
loss_diff = torch.sum((gt_loss - my_loss) ** 2)
test_error(loss_diff, test="yolo")    