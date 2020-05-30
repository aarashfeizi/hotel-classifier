import logging
import sys

from torch.utils.data import DataLoader
from torchvision import transforms

import model_helper_functions
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

    train_set = None
    test_set = None
    val_set = None

    # train_classification_dataset = CUBClassification(args, transform=data_transforms, mode='train')

    if args.dataset_name == 'cub':

        if args.dataset_split_type == 'original':
            train_set = CUBTrain_Top(args, transform=data_transforms)
            val_set = CUBTest_Fewshot(args, transform=data_transforms, mode='val')
            test_set = CUBTest_Fewshot(args, transform=data_transforms)

        elif args.dataset_split_type == 'new':  # mode = [knwn_cls_test, knwn_cls_val, train, uknwn_cls_test, uknwn_cls_val]
            train_set = CUBTrain_Top(args, transform=data_transforms, mode='train')
            val_set_known = CUBTest_Fewshot(args, transform=data_transforms, mode='val_seen')
            test_set_known = CUBTest_Fewshot(args, transform=data_transforms, mode='test_seen')
            val_set_unknown = CUBTest_Fewshot(args, transform=data_transforms, mode='val_unseen')
            test_set_unknown = CUBTest_Fewshot(args, transform=data_transforms, mode='test_unseen')

    elif args.dataset_name == 'omniglot':
        train_set = OmniglotTrain(args, transform=data_transforms)
        # val_set = CUBTest(args, transform=data_transforms, mode='val')
        test_set = OmniglotTest(args, transform=transforms.ToTensor())
    elif args.dataset_name == 'hotels':

        train_set = HotelTrain(args, transform=data_transforms, mode='train')
        val_set_known = HotelTest(args, transform=data_transforms, mode='val_seen')
        val_set_unknown = HotelTest(args, transform=data_transforms, mode='val_unseen')

        if args.test:
            test_set_known = HotelTest(args, transform=data_transforms, mode='test_seen')
            test_set_unknown = HotelTest(args, transform=data_transforms, mode='test_unseen')

    else:
        print('Fuck: ', args.dataset_name)

    print('way:', args.way)

    # train_classify_loader = DataLoader(train_classification_dataset, batch_size=args.batch_size, shuffle=False,
    #                                    num_workers=args.workers)
    test_loaders = []
    val_loaders = []

    if args.test:
        if args.dataset_split_type == 'original':
            test_loaders.append(DataLoader(test_set, batch_size=args.way, shuffle=False, num_workers=args.workers))

        elif args.dataset_split_type == 'new':
            test_loaders.append(
                DataLoader(test_set_known, batch_size=args.way, shuffle=False, num_workers=args.workers))
            test_loaders.append(
                DataLoader(test_set_unknown, batch_size=args.way, shuffle=False, num_workers=args.workers))
    ###
    import time
    import multiprocessing
    use_cuda = torch.cuda.is_available()
    core_number = multiprocessing.cpu_count()
    batch_size = 64
    best_num_worker = [0, 0]
    best_time = [99999999, 99999999]
    print('cpu_count =', core_number)

    def loading_time(num_workers, pin_memory):
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        start = time.time()
        for epoch in range(4):
            for batch_idx, (_, _, _) in enumerate(train_loader):
                if batch_idx == 15:
                    break
                pass
        end = time.time()
        print("  Used {} second with num_workers = {}".format(end - start, num_workers))
        return end - start

    for pin_memory in [False, True]:
        print("While pin_memory =", pin_memory)
        for num_workers in range(0, core_number * 2 + 1, 4):
            current_time = loading_time(num_workers, pin_memory)
            if current_time < best_time[pin_memory]:
                best_time[pin_memory] = current_time
                best_num_worker[pin_memory] = num_workers
            else:  # assuming its a convex function
                if best_num_worker[pin_memory] == 0:
                    the_range = []
                else:
                    the_range = list(range(best_num_worker[pin_memory] - 3, best_num_worker[pin_memory]))
                for num_workers in (
                        the_range + list(range(best_num_worker[pin_memory] + 1, best_num_worker[pin_memory] + 4))):
                    current_time = loading_time(num_workers, pin_memory)
                    if current_time < best_time[pin_memory]:
                        best_time[pin_memory] = current_time
                        best_num_worker[pin_memory] = num_workers
                break
    if best_time[0] < best_time[1]:
        print("Best num_workers =", best_num_worker[0], "with pin_memory = False")
        workers = best_num_worker[0]
        pin_memory = False
    else:
        print("Best num_workers =", best_num_worker[1], "with pin_memory = True")
        workers = best_num_worker[1]
        pin_memory = True
    ###
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                              pin_memory=pin_memory)

    if (val_set is not None) or (val_set_known is not None):

        if args.dataset_split_type == 'original':
            val_loaders.append(DataLoader(val_set, batch_size=args.way, shuffle=False, num_workers=workers,
                              pin_memory=pin_memory))

        elif args.dataset_split_type == 'new':
            val_loaders.append(
                DataLoader(val_set_known, batch_size=args.way, shuffle=False, num_workers=workers,
                              pin_memory=pin_memory))
            val_loaders.append(
                DataLoader(val_set_unknown, batch_size=args.way, shuffle=False, num_workers=workers,
                              pin_memory=pin_memory))
    else:
        val_loaders = None
        raise Exception('No validation data is set!')

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # train resnet

    num_classes = train_set.num_classes
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
        tm_net, best_model_top = model_methods_top.train_fewshot(tm_net, loss_fn, args, train_loader, val_loaders)
    else:  # test
        logger.info('Testing')
        best_model_top = args.model_name

    # testing
    if args.test:
        logger.info(f"Loading {best_model_top} model...")
        tm_net = model_methods_top.load_model(args, tm_net, best_model_top)

        if args.dataset_split_type == 'new':
            model_methods_top.test_fewshot(args, tm_net, test_loaders[0], loss_fn, comment='known')
            model_methods_top.test_fewshot(args, tm_net, test_loaders[1], loss_fn, comment='unknown')
        else:  # original
            model_methods_top.test_fewshot(args, tm_net, test_loaders[0], loss_fn)
    else:
        logger.info("NO TESTING DONE.")
    #  learning_rate = learning_rate * 0.95


if __name__ == '__main__':
    main()
