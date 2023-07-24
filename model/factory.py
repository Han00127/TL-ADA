"""
Factory method for easily getting model by name.
written by wgchang
"""
from model import resnet
from model import lenet
__sets = {}

for model_name in ['resnet18','resnet34', 'resnet50', 'p_resnet50','resnet101',  'resnet152', 'lenet']:

    if model_name in ['resnet18','resnet34','resnet50', 'p_resnet50','resnet101', 'resnet152']:
        eval_str = "resnet.{}"
        __sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains,
                                     model_name=model_name, eval_str=eval_str:
                              eval(eval_str.format(model_name))(pretrained=pretrained, num_classes=num_classes,
                                                                in_features=in_features, num_domains=num_domains))
    elif model_name == 'lenet':
        __sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains:
                              lenet.lenet(num_classes=num_classes, in_features=in_features,
                                              weights_init_path=None, num_domains=num_domains))

def get_model(model_name, num_classes, in_features=0, num_domains=2, pretrained=False):
    model_key = model_name
    if model_key not in __sets:
        raise KeyError(
            'Unknown Model: {}, num_classes: {}, in_features: {}'.format(model_key, num_classes, in_features))
    return __sets[model_key](num_classes=num_classes, in_features=in_features,
                             pretrained=pretrained, num_domains=num_domains)
