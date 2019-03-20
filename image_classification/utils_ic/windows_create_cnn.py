# python regular libraries
from string import digits

# fast.ai
from fastai.vision import Callable, DataBunch, nn, \
    Optional, create_body, Collection, Union, Floats, \
    SplitFuncOrIdxList, Any, Learner, create_head, ifnone, apply_init


def _default_split(m: nn.Module): return (m[1], )


def _resnet_split(m: nn.Module): return (m[0][6], m[1])


def _squeezenet_split(m: nn.Module): return (m[0][0][5], m[0][0][8], m[1])


def _densenet_split(m: nn.Module): return (m[0][0][7], m[1])


def _vgg_split(m: nn.Module): return (m[0][0][22], m[1])


def _alexnet_split(m: nn.Module): return (m[0][0][6], m[1])


def appropriate_nbr_features():
    nbr_features_dict = dict()

    nbr_features_dict['alexnet'] = 512

    nbr_features_dict['vgg16'] = nbr_features_dict['vgg19'] = \
        nbr_features_dict['vgg16_bn'] = nbr_features_dict['vgg19_bn'] = \
        nbr_features_dict['resnet18'] = nbr_features_dict['resnet34'] = \
        nbr_features_dict['squeezenet1_0'] = \
        nbr_features_dict['squeezenet1_1'] = 1024

    nbr_features_dict['resnet50'] = nbr_features_dict['resnet101'] = \
        nbr_features_dict['resnet152'] = 4096

    nbr_features_dict['densenet121'] = 32

    nbr_features_dict['densenet169'] = nbr_features_dict['densenet201'] = 64

    return nbr_features_dict


global number_features

number_features = appropriate_nbr_features()


def custom_create_cnn(data: DataBunch, arch: Callable,
                      cut: Union[int, Callable] = None,
                      pretrained: bool = True,
                      lin_ftrs: Optional[Collection[int]] = None,
                      ps: Floats = 0.5,
                      custom_head: Optional[nn.Module] = None,
                      split_on: Optional[SplitFuncOrIdxList] = None,
                      bn_final: bool = False, **learn_kwargs: Any)->Learner:

    # Inspired from:
    # https://forums.fast.ai/t/lesson-1-notebook-stuck-in-create-cnn/37486/5
    # Thank you ZhekaMeka!

    model_name = arch.__name__
    if model_name not in number_features:
        raise Exception("{} is currently not supported. Please use any of: {}"
                        .format(model_name, list(number_features.keys())))

    num_features = number_features[model_name]
    body = create_body(arch, pretrained, cut)
    head = custom_head or create_head(num_features, data.c, lin_ftrs,
                                      ps=ps, bn_final=bn_final)
    model = nn.Sequential(body, head)
    learn = Learner(data, model, **learn_kwargs)

    # extraction of model name
    needed_split = "_{}_split".format(model_name)
    # removal of potential digits (e.g. as in vgg16 or resnet50)
    remove_digits = str.maketrans('', '', digits)
    needed_split = needed_split.translate(remove_digits).replace('__', '_')\
        .replace('_bn', '')

    print("Model *{}* is using {} features".format(model_name, num_features))

    learn.split(ifnone(split_on, eval(needed_split)))

    if pretrained:
        learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)
    return learn
