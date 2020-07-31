import logging
import sys

from torch.utils.data import DataLoader
from torchvision import transforms

import model_helper_functions
import utils
from cub_dataloader import *
from hotel_dataloader import *
from models.top_model import *
from omniglot_dataloader import *
from losses import TripletLoss


###
# todo for next week

# fuckin overleaf
# Average per class for metrics (k@n) ???
# k@n for training
# random crop in train
# visualize images before and after transformation to see what information is lost
# do NOT transform scale for validation


def _logger():
    logging.basicConfig(format='%(asctime)s - %(message)s', stream=sys.stdout, level=logging.INFO)
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

    data_transforms_train, transform_list_train = utils.TransformLoader(image_size,
                                                                        rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=True)

    logger.info(f'train transforms: {transform_list_train}')

    data_transforms_val, transform_list_val = utils.TransformLoader(image_size,
                                                                    rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=False)
    logger.info(f'val transforms: {transform_list_val}')

    if args.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print("use gpu:", args.gpu_ids, "to train.")

    train_set = None
    test_set = None
    val_set = None
    val_set_known_fewshot = None
    val_set_unknown_fewshot = None

    # train_classification_dataset = CUBClassification(args, transform=data_transforms, mode='train')

    if args.dataset_name == 'cub':

        if args.dataset_split_type == 'original':
            train_set = CUBTrain_Top(args, transform=data_transforms_train)
            val_set = CUBTest_Fewshot(args, transform=data_transforms_val, mode='val')
            test_set = CUBTest_Fewshot(args, transform=data_transforms_val)

        elif args.dataset_split_type == 'new':  # mode = [knwn_cls_test, knwn_cls_val, train, uknwn_cls_test, uknwn_cls_val]
            train_set = CUBTrain_Top(args, transform=data_transforms_train, mode='train')
            val_set_known_fewshot = CUBTest_Fewshot(args, transform=data_transforms_val, mode='val_seen')
            test_set_known = CUBTest_Fewshot(args, transform=data_transforms_val, mode='test_seen')
            val_set_unknown_fewshot = CUBTest_Fewshot(args, transform=data_transforms_val, mode='val_unseen')
            test_set_unknown = CUBTest_Fewshot(args, transform=data_transforms_val, mode='test_unseen')

    elif args.dataset_name == 'omniglot':
        train_set = OmniglotTrain(args, transform=data_transforms_train)
        # val_set = CUBTest(args, transform=data_transforms, mode='val')
        test_set = OmniglotTest(args, transform=transforms.ToTensor())
    elif args.dataset_name == 'hotels':

        print('*' * 10)
        if args.metric_learning:
            train_set = HotelTrain_Metric(args, transform=data_transforms_train, mode='train', save_pictures=False)
            print('*' * 10)
            val_set_known_metric = HotelTrain_Metric(args, transform=data_transforms_val, mode='val_seen',
                                                     save_pictures=False)
            print('*' * 10)
            val_set_unknown_metric = HotelTrain_Metric(args, transform=data_transforms_val, mode='val_unseen',
                                                       save_pictures=False)

        else:
            train_set = HotelTrain_FewShot(args, transform=data_transforms_train, mode='train', save_pictures=False)
            print('*' * 10)

        val_set_known_fewshot = HotelTest(args, transform=data_transforms_val, mode='val_seen', save_pictures=False)
        print('*' * 10)
        val_set_unknown_fewshot = HotelTest(args, transform=data_transforms_val, mode='val_unseen', save_pictures=False)

        if args.test:
            test_set_known = HotelTest(args, transform=data_transforms_val, mode='test_seen')
            print('*' * 10)
            test_set_unknown = HotelTest(args, transform=data_transforms_val, mode='test_unseen')
            print('*' * 10)

            # todo test not supported for metric learning

        if args.cbir:
            db_set = Hotel_DB(args, transform=data_transforms_val, mode='val')
            db_set_train = Hotel_DB(args, transform=data_transforms_val, mode='train')

    else:
        logger.error(f'Dataset not suppored:  {args.dataset_name}')

    logger.info(f'few shot evaluation way: {args.way}')

    # train_classify_loader = DataLoader(train_classification_dataset, batch_size=args.batch_size, shuffle=False,
    #                                    num_workers=args.workers)
    test_loaders = []

    if args.test:
        if args.dataset_split_type == 'original':
            test_loaders.append(DataLoader(test_set, batch_size=args.way, shuffle=False, num_workers=args.workers))

        elif args.dataset_split_type == 'new':
            test_loaders.append(
                DataLoader(test_set_known, batch_size=args.way, shuffle=False, num_workers=args.workers))
            test_loaders.append(
                DataLoader(test_set_unknown, batch_size=args.way, shuffle=False, num_workers=args.workers))

    # workers = 4
    # pin_memory = False
    if args.find_best_workers:
        workers, pin_memory = utils.get_best_workers_pinmemory(args, train_set,
                                                               pin_memories=[True],
                                                               starting_from=0,
                                                               logger=logger)
    else:
        workers = args.workers
        pin_memory = args.pin_memory

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                              pin_memory=pin_memory)

    val_loaders_fewshot = utils.get_val_loaders(args, val_set, val_set_known_fewshot, val_set_unknown_fewshot, workers,
                                                pin_memory)

    if args.metric_learning:
        val_loaders_metric = utils.get_val_loaders(args, val_set, val_set_known_metric, val_set_unknown_metric, workers,
                                                   pin_memory)

    if args.cbir:
        db_loader = DataLoader(db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                               pin_memory=pin_memory)

        db_loader_train = DataLoader(db_set_train, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                                     pin_memory=pin_memory)

    if args.loss == 'bce':
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif args.loss == 'trpl':
        loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_fn = TripletLoss(margin=args.margin)
    else:
        raise Exception('Loss function not supported: ' + args.loss)

    num_classes = train_set.num_classes
    logger.info(f'Num classes in train: {num_classes}')

    model_methods_top = model_helper_functions.ModelMethods(args, logger, 'top')
    tm_net = top_module(args=args, num_classes=num_classes)

    print(model_methods_top.save_path)

    # multi gpu
    if len(args.gpu_ids.split(",")) > 1:
        tm_net = torch.nn.DataParallel(tm_net)

    if args.cuda:
        tm_net.cuda()

    logger.info('Training Top')
    if args.model_name == '':
        logger.info('Training')
        if args.metric_learning:
            tm_net, best_model_top = model_methods_top.train_metriclearning(tm_net, loss_fn, loss_fn_bce, args,
                                                                            train_loader, val_loaders_metric,
                                                                            val_loaders_fewshot)
        else:
            tm_net, best_model_top = model_methods_top.train_fewshot(tm_net, loss_fn, args, train_loader,
                                                                     val_loaders_fewshot)
        logger.info('Calculating K@Ns for Validation')

        model_methods_top.make_emb_db(args, tm_net, db_loader_train,
                                      eval_sampled=False,
                                      eval_per_class=True, newly_trained=True,
                                      batch_size=args.db_batch,
                                      mode='train_sampled')

        model_methods_top.make_emb_db(args, tm_net, db_loader,
                                      eval_sampled=args.sampled_results,
                                      eval_per_class=args.per_class_results, newly_trained=True,
                                      batch_size=args.db_batch,
                                      mode='val')
    else:  # test
        logger.info('Testing')
        best_model_top = args.model_name

    if args.katn and args.model_name != '':
        logger.info(f"Not training, loading {best_model_top} model...")
        tm_net = model_methods_top.load_model(args, tm_net, best_model_top)
        logger.info('Calculating K@Ns for Validation')
        model_methods_top.make_emb_db(args, tm_net, db_loader_train,
                                      eval_sampled=False,
                                      eval_per_class=True, newly_trained=True,
                                      batch_size=args.db_batch,
                                      mode='train_sampled')
        model_methods_top.make_emb_db(args, tm_net, db_loader,
                                      eval_sampled=args.sampled_results,
                                      eval_per_class=args.per_class_results, newly_trained=False,
                                      batch_size=args.db_batch,
                                      mode='val')

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
