import logging
import sys
from comet_ml import Experiment

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from dataloader import *
from models.top_model import *


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

    if args.dataset_name == 'cub':
        trainSet = CUBTrain(args, transform=data_transforms)
        valSet = CUBTest(args, transform=data_transforms, mode='val')
        testSet = CUBTest(args, transform=data_transforms)
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

    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)

    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if valSet is not None:
        valLoader = DataLoader(valSet, batch_size=args.way, shuffle=False, num_workers=args.workers)
    else:
        valLoader = testLoader

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    tm_net = top_module(args=args)

    model_methods = utils.ModelMethods(args)

    print(model_methods.save_path)

    # multi gpu
    if len(args.gpu_ids.split(",")) > 1:
        tm_net = torch.nn.DataParallel(tm_net)
    #
    # import pdb
    # pdb.set_trace()

    if args.cuda:
        tm_net.cuda()

    if args.model_name == '':  # train
        experiment = Experiment(api_key="y7W3nBB2KpTXelJAzRFwK0mqn",
                                project_name="hotels", workspace="aarashfeizi")
        logger.info('Training')
        tm_net, best_model = model_methods.train(tm_net, loss_fn, args, trainLoader, valLoader, logger)
    else:  # test
        logger.info('Testing')
        best_model = args.model_name

    # testing
    logger.info(f"Loading {best_model} model...")
    tm_net = model_methods.load_model(args, tm_net, best_model, logger)

    tm_net.eval()

    tests_right, tests_error = 0, 0

    for _, (test1, test2) in enumerate(testLoader, 1):
        if args.cuda:
            test1, test2 = test1.cuda(), test2.cuda()
        test1, test2 = Variable(test1), Variable(test2)
        output = tm_net.forward(test1, test2).data.cpu().numpy()
        pred = np.argmax(output)
        if pred == 0:
            tests_right += 1
        else:
            tests_error += 1

    test_acc = tests_right * 1.0 / (tests_right + tests_error)
    logger.info('$' * 70)
    logger.info(
        'TEST:\tTest set\tcorrect:\t%d\terror:\t%d\ttest_acc:%f\t' % (tests_right, tests_error, test_acc))
    logger.info('$' * 70)

    #  learning_rate = learning_rate * 0.95


if __name__ == '__main__':
    main()
