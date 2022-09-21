from collections import defaultdict

from utils import weight_test_loss
import numpy as np


def print_stats(args, num_step, test_clients, val_metrics, logger, writer, ret_score='miou', action=None,
                server=None, train_clients=None, label2color=None, denorm=None, last=False):

    def log_samples(ret_samples, sample_type, silobn_type=''):
        for k, (img, target, pred) in enumerate(ret_samples):
            img = (denorm(img) * 255).astype(np.uint8)
            target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
            pred = label2color(pred).transpose(2, 0, 1).astype(np.uint8)

            concat_img = np.concatenate((img, target, pred), axis=2)  # concat along width
            prt_im = concat_img.transpose(1, 2, 0)

            logger.log_image(f'Sample_{k}_test_{sample_type}{silobn_type}', [prt_im],
                             caption=["RGB, Target, Prediction"])

    def log(score, loss, test_type='eval', test_client='', silobn_type='', last=False):
        last = 'Last_' if last else ''
        logger.log_metrics({
            f'{last}{test_type}{test_client}{silobn_type}_Loss': loss,
            f'{last}{test_type}{test_client}{silobn_type}_Overall_Acc': score['Overall Acc'],
            f'{last}{test_type}{test_client}{silobn_type}_MeanIoU': score['Mean IoU']}, step=num_step)
        # wandb table
        columns = ['class', 'IoU']
        data = [[k, v] for k, v in score['Class IoU'].items()]
        data = [[x[0], x[1]] if type(x[1]) != str else [x[0], -1.0] for x in data]
        logger.log_table(test_type + test_client + "_Class_IoU", columns, data, step=num_step)
        columns = ['class', 'Acc']
        data = [[k, v] for k, v in score['Class Acc'].items()]
        data = [[x[0], x[1]] if type(x[1]) != str else [x[0], -1.0] for x in data]
        logger.log_table(test_type + test_client + "_Class_Acc", columns, data, step=num_step)

    def make_fed_test(ref_clients, test_type='eval', last=False):
        writer.write("Evaluating on the whole training set...") if test_type == 'eval' else writer.write("Testing...")
        if test_type == 'eval':
            for c in ref_clients:
                c.dataset.test_bisenetv2 = True
        scores = defaultdict(lambda: defaultdict(lambda: {}))
        if args.dataset == "idda" and test_type == 'test':  # two different tests
            for c in ref_clients:
                if args.algorithm == 'SiloBN' and str(c) == 'test_user_same_domain':
                    if args.clients_type == 'heterogeneous':
                        silobn_tests = ['_standard', '_by_domain']
                    else:
                        raise NotImplementedError
                else:
                    silobn_tests = ['']
                ret_samples_bool = False
                if num_step == args.num_rounds and args.save_samples:
                    ret_samples_bool = True
                client_to_test = [c]
                for silobn_type in silobn_tests:
                    losses, ret_samples = server.test_model(client_to_test, val_metrics, ret_samples_bool,
                                                            silobn_type=silobn_type)
                    score = val_metrics.get_results()
                    loss = weight_test_loss(losses)
                    test_user = '_diff_dom' if c.id == 'test_user_diff_domain' else '_same_dom'
                    if ret_samples_bool:
                        log_samples(ret_samples, sample_type=test_user, silobn_type=silobn_type)
                    log(score, loss, test_type='Test', test_client=test_user, silobn_type=silobn_type, last=last)
                    scores[str(c)][silobn_type]['acc'] = score['Overall Acc']
                    scores[str(c)][silobn_type]['mIoU'] = score['Mean IoU']
        else:
            ret_samples_bool = False
            if num_step == args.num_rounds and test_type == 'test' and args.save_samples:
                ret_samples_bool = True
            losses, ret_samples = server.test_model(ref_clients, val_metrics, ret_samples_bool)
            score = val_metrics.get_results()
            loss = weight_test_loss(losses)

            if test_type == 'eval':
                log(score, loss, test_type='Tot_Train', last=last)
            else:
                log(score, loss, test_type='test', last=last)

            if ret_samples_bool:
                log_samples(ret_samples, sample_type='')
            if test_type == 'eval':
                for c in ref_clients:
                    c.dataset.test_bisenetv2 = False
            scores['test_user']['']['acc'] = score['Overall Acc']
            scores['test_user']['']['mIoU'] = score['Mean IoU']

        scores = dict(scores)
        for k in scores.keys():
            scores[k] = dict(scores[k])
        return scores

    def perform_test(c):
        ret_samples_ids = None
        if num_step == args.num_epochs and args.save_samples:
            ret_samples_ids = np.random.choice(len(c.loader), 3, replace=False)
        loss, ret_samples = c.test(val_metrics, ret_samples_ids)
        score = val_metrics.get_results()
        val_metrics.reset()
        return loss, ret_samples, ret_samples_ids, score

    def make_centr_test():
        writer.write("Testing...")
        if args.dataset == "idda":  # two different test
            scores = []
            for c in test_clients:
                loss, ret_samples, ret_samples_ids, score = perform_test(c)
                test_user = '_diff_dom' if c.id == 'test_user_diff_domain' else '_same_dom'
                if ret_samples_ids is not None:
                    log_samples(ret_samples, sample_type=test_user)
                log(score, loss, test_type='Test', test_client=test_user)
                scores.append(score['Overall Acc']) if ret_score == 'acc' else scores.append(score['Mean IoU'])
            return scores
        else:
            client_to_test = test_clients[0]
            loss, ret_samples, ret_samples_ids, score = perform_test(client_to_test)
            log(score, loss, test_type='Test')
            if ret_samples_ids is not None:
                log_samples(ret_samples, sample_type='')
            if ret_score == 'acc':
                return score['Overall Acc']
            return score['Mean IoU']

    test_type = 'eval' if action == 'eval' else 'test'
    clients_to_test = train_clients if action == 'eval' else test_clients

    if args.framework == 'federated':
        return make_fed_test(clients_to_test, test_type=test_type, last=last)
    return make_centr_test()
