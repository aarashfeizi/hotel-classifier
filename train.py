import logging
import sys

from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from dataloader import *
from models.top_model import *
import model_helper_functions


def _logger():
    logging.basicConfig(format='%(message)s', stream=sys.stdout, level=logging.INFO)
    return logging.getLogger()


def main():
    logger = _logger()

    args = utils.get_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_size = -1

    if args.image_size > 0:
        image_size = args.image_size
    elif args.dataset_name == 'cub':
        image_size = 224
    elif args.dataset_name == 'omniglot':
        image_size = 105
    elif args.dataset_name == 'hotels':
        image_size = 300

    data_transforms = utils.TransformLoader(image_size, rotate=args.rotate).get_composed_transform(aug=args.aug)

    # data_transforms = transforms.Compose([
    #     transforms.Resize([int(image_size), int(image_size)]),
    #     transforms.RandomAffine(15),
    #     transforms.ToTensor()
    # ])

    # train_dataset = dset.ImageFolder(root=Flags.train_path)
    # test_dataset = dset.ImageFolder(root=Flags.test_path)

    if args.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print("use gpu:", args.gpu_ids, "to train.")

    trainSet = None
    testSet = None
    valSet = None

    train_classification_dataset = CUBClassification(args, transform=data_transforms, mode='train')

    if args.dataset_name == 'cub':
        trainSet = CUBTrain_Top(args, transform=data_transforms)
        valSet = CUBTest_Fewshot(args, transform=data_transforms, mode='val')
        testSet = CUBTest_Fewshot(args, transform=data_transforms)
    elif args.dataset_name == 'omniglot':
        trainSet = OmniglotTrain(args, transform=data_transforms)
        # valSet = CUBTest(args, transform=data_transforms, mode='val')
        testSet = OmniglotTest(args, transform=transforms.ToTensor())
    elif args.dataset_name == 'hotels':
        trainSet = HotelTrain(args, transform=data_transforms)
        # valSet = CUBTest(args, transform=data_transforms, mode='val')
        testSet = HotelTest(args, transform=data_transforms)
    else:
        print('Fuck: ', args.dataset_name)

    print('way:', args.way)

    train_classify_loader = DataLoader(train_classification_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.workers)

    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)

    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if valSet is not None:
        valLoader = DataLoader(valSet, batch_size=args.way, shuffle=False, num_workers=args.workers)
    else:
        valLoader = testLoader

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # train resnet

    num_classes = train_classification_dataset.num_classes
    #
    # print('num_classes', num_classes)
    #
    # feat_ext = resnet18(pretrained=True, num_classes=num_classes)
    #
    # if len(args.gpu_ids.split(",")) > 1:
    #     feat_ext = torch.nn.DataParallel(feat_ext)
    #
    # if args.cuda:
    #     feat_ext.cuda()
    #
    # model_methods = model_helper_functions.ModelMethods(args, logger, 'res')
    #
    # logger.info('Training Res')
    # feat_net = model_methods.train_classify(feat_ext, loss_fn, args, train_classify_loader, None)
    #
    # ################################
    #
    # print('loading trained feature model')
    # feat_net = model_methods.load_model(args, feat_net, best_res_model)
    #
    # print('loading trained feature model done!')

    model_methods_top = model_helper_functions.ModelMethods(args, logger, 'top')
    # tm_net = top_module(args=args, trained_feat_net=feat_net, num_classes=num_classes)
    tm_net = top_module(args=args, num_classes=num_classes)

    print(model_methods_top.save_path)

    # multi gpu
    if len(args.gpu_ids.split(",")) > 1:
        tm_net = torch.nn.DataParallel(tm_net)
    #
    # import pdb
    # pdb.set_trace()

    if args.cuda:
        tm_net.cuda()

    logger.info('Training Top')
    if args.model_name == '':  # train
        logger.info('Training')
        tm_net, best_model_top = model_methods_top.train_fewshot(tm_net, loss_fn, args, trainLoader, valLoader)
    else:  # test
        logger.info('Testing')
        best_model_top = args.model_name

    # testing
    logger.info(f"Loading {best_model_top} model...")
    tm_net = model_methods_top.load_model(args, tm_net, best_model_top)

    model_methods_top.test_fewshot(args, tm_net, testLoader)

    #  learning_rate = learning_rate * 0.95


if __name__ == '__main__':
    main()
