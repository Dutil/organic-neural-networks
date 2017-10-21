import sys

from collections import OrderedDict
import yaml
import itertools
import argparse
import numpy as np
import os
import theano
import theano.tensor as T
import util
import activation
import initialization
import steprules
import whitening
import dataset_loader

def build_prong(input_dim, n_outputs, layers, hyperparameters):

    dims = (input_dim,) + layers + (n_outputs,)

    layers = [
        dict(f=activation.tanh,
             c=util.shared_floatx((m,),   initialization.constant(0)),   # input mean
             U=util.shared_floatx((m, m), initialization.identity()),    # input whitening matrix
             W=util.shared_floatx((m, n), initialization.orthogonal()),  # weight matrix
             g=util.shared_floatx((m if hyperparameters['bn_before'] else n,),   initialization.constant(1)),   # gammas (for batch normalization)
             b=util.shared_floatx((n,),   initialization.constant(0)))   # bias
        for m, n in util.safezip(dims[:-1], dims[1:])]
    layers[-1]["f"] = activation.logsoftmax

    # Get the total number of parameters:
    nb_param = [m * n for m, n in util.safezip(dims[:-1], dims[1:])]
    nb_param = sum(nb_param)

    return layers, nb_param

def off_diagonal_corr_penalty(a, hyperparameters):

    cov = T.dot(a.T, a) / a.shape[0]

    if not hyperparameters['bn_before']: # If not centered
        expectation = a.mean(axis=0)
        cov -= T.outer(expectation, expectation)

    diag = T.diag(cov)
    pure_cov = cov - T.diag(diag)

    cov_cost = (pure_cov * pure_cov).mean() * cov.shape[0]/(cov.shape[0] + 1)

    return cov_cost

def diagonal_corr_penalty(a, hyperparameters):

    variance = (a * a).mean(0)

    if not hyperparameters['bn_before']: # If not centered
        expectation = a.mean(axis=0)
        variance -= expectation * expectation

    variance_diff = variance - 1.
    cost = (variance_diff * variance_diff).mean()

    return cost


def get_correlation_penalty(a, hyperparameters):

    cov = T.dot(a.T, a)/a.shape[0]

    if hyperparameters['bn_normalize'] and hyperparameters['bn_before'] and not hyperparameters['bn_scale']:
        # The activation are centured and normalized
        cov_minus_diag = cov * (1.-T.identity_like(cov))
        cov_cost = (cov_minus_diag * cov_minus_diag).mean()
    else:
        expectation = a.mean(axis=0)
        cov -= T.outer(expectation, expectation)
        diag = T.diag(cov)
        diag_diff = diag - 1.
        pure_cov = cov - T.diag(diag)
        cov_cost = (pure_cov * pure_cov).mean()
        cov_cost += cov.shape[0] * (diag_diff * diag_diff).mean()
        cov_cost *= cov.shape[0]/(cov.shape[0] + 1)

    return hyperparameters['correlation_penalty'] * cov_cost

def get_updates(layers, h, hyperparameters):


    # reparametrization updates
    reparameterization_updates = []
    # theano graphs with assertions & breakpoints, to be evaluated after
    # performing the updates
    reparameterization_checks = []

    # correlation penlaty etc.
    off_diag_cost = 0.
    diag_cost = 0.


    for i, layer in enumerate(layers):
        f, c, U, W, g, b = [layer[k] for k in "fcUWgb"]

        # construct reparameterization graph
        if hyperparameters['whiten_weights']:
            updates, checks = whitening.get_updates(
                h, c, U, V=W, d=b,
                decomposition="svd", zca=True, bias=hyperparameters['eigenvalue_bias'])
            reparameterization_updates.extend(updates)
            reparameterization_checks.extend(checks)

            # whiten input
            h = T.dot(h - c, U)


        # Compute the batch norm before or after the linear transformation
        if hyperparameters['bn_before']:

            h -= h.mean(axis=0, keepdims=True)
            if hyperparameters['bn_normalize']:
                h /= T.sqrt(h.var(axis=0, keepdims=True) + hyperparameters["variance_bias"])

            if hyperparameters['bn_scale']:
                h *= g

        off_diag_cost += off_diagonal_corr_penalty(h, hyperparameters)
        diag_cost += diagonal_corr_penalty(h, hyperparameters)

        # compute layer as usual
        h = T.dot(h, W)

        if hyperparameters['batch_normalize'] and not hyperparameters['bn_before']:
            h -= h.mean(axis=0, keepdims=True)
            h /= T.sqrt(h.var(axis=0, keepdims=True) + hyperparameters["variance_bias"])
            h *= g# TODO have another g
        h += b
        h = f(h)

    return reparameterization_updates, reparameterization_checks, h, off_diag_cost, diag_cost

def get_fisher(parameters_by_layer, n_outputs, logp, cross_entropy, hyperparameters, parameters_index):

    fisher = None

    def estimate_fisher(outputs, n_outputs, parameters):

        # shape (sample_size, n_outputs, #parameters)
        grads = T.stack(*[util.batched_flatcat(
            T.jacobian(outputs[:, j], parameters))
            for j in xrange(n_outputs)])
        # ravel the batch and output axes so that the product will sum
        # over the outputs *and* over the batch. divide by the batch
        # size to get the batch mean.
        # We get a subset, otherwise everything explose
        grads = grads.reshape((grads.shape[0] * grads.shape[1], grads.shape[2]))[:, parameters_index]
        fisher = T.dot(grads.T, grads) / grads.shape[0]
        return fisher

    # estimate_fisher will perform one backpropagation per sample, so
    # don't go wild
    fisher_parameters = [p for layer in parameters_by_layer for p in layer]

    # TODO: to remove
    if hyperparameters["objective"] == "loss":
        # fisher on loss
        fisher = estimate_fisher(cross_entropy[:, np.newaxis],
                                 1, fisher_parameters)
    elif hyperparameters["objective"] == "output":
        # fisher on output
        fisher = estimate_fisher(logp[:, :],
                                 n_outputs, fisher_parameters)

    return fisher

def build_parser():
    parser = argparse.ArgumentParser(
        description="Model for Natural Gradient descent")

    # Model specific options
    parser.add_argument('--layers', nargs='+', default=[500, 300, 100], type=int, help="The layers of the mlp.")
    parser.add_argument('--epochs', default=100, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--batch_size', default=100, type=int, help="The batch size.")
    parser.add_argument('--folder', default='./', help='The folder where to store the experiments. Will be created if not already exists.')
    parser.add_argument('--bn', dest='bn', action='store_true', default=False, help="If we do batch norm or not.")
    parser.add_argument('--fisher', action='store_true',
                        default=False,  help='If we want to save the fisher information matrix or not.')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--sample-size', default=10, type=int, help='The number of minibatch used to compute the FIM.')
    parser.add_argument('--interval', default=100, type=int, help='After how many minibatch we want to reparametrize. If = 0, No reparametrization is done (SGD).')
    parser.add_argument('--fisher-dimension', dest='fisher_dimension', default=1000, type=int, help='The number of dimensions to keep to compute the FIM.')
    parser.add_argument('--whiten-inputs', dest='whiten', action='store_true', default=False, help='If we want to whiten the inputs with zca or not.')
    parser.add_argument('--whiten-weights', action='store_true', default=False, help='If we want to whiten the weights with zca or not.')

    parser.add_argument('--bn-before', dest='bn_before', action='store_true', default=False, help='If we want to do batch norm before the linear transformation,')
    parser.add_argument('--bn-normalize', action='store_true',
                        default=False, help='If we want to normalize the activation (right after the activation),')
    parser.add_argument('--bn-scale', action='store_true',
                    default=False, help='If we want to scale the activation after the centering and normalization for the activation.')

    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='If we want different parameters for the FIM after each epochs.')
    parser.add_argument('--estimation-size', default=1000, type=int, help='Number of examples used to do the repametrization.')

    parser.add_argument('--eigenvalue-bias', dest='eigenvalue_bias', type=float, default=1e-3, help='the bias to add to the zca transformation.')
    parser.add_argument('--variance-bias', dest='variance_bias', type=float, default=1e-8, help='The bias to the variance for batch norm.')


    parser.add_argument('--data-folder', default='/u/dutilfra/datasets/', help='The folder contening the dataset.')
    parser.add_argument('--dataset-name', default='mnist', help='Which dataset to use.')
    # TODO: spliter la cost, avoir un parametre pour quand n veux utiliser la
    # meme.
    parser.add_argument('--off-diagonal-penalty', type=float, default=0.,
                        help='The correlation penalty on the off diagonal elements')

    parser.add_argument('--diagonal-penalty', type=float, default=0.,
                        help='The penalty to apply on the diagonal of the correlation matrix')

    parser.add_argument('--share-correlation-penalty', action='store_true', default=False,
                        help='If we want to share the same correlation penalty for off and on diagonal.')


    return parser

def parse_args(argv):
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)
    
    opt.layers = tuple(opt.layers)
    layers = opt.layers
    nepochs = opt.epochs
    batch_size = opt.batch_size
    folder = opt.folder
    batch_normalize = opt.bn
    save_fisher = opt.fisher
    lr = opt.lr
    sample_size = opt.sample_size
    interval = opt.interval
    whiten_inputs = opt.whiten
    fisher_dimension = opt.fisher_dimension
    shuffle = opt.shuffle
    eigenvalue_bias = opt.eigenvalue_bias
    variance_bias = opt.variance_bias
    estimation_size = opt.estimation_size

    data_folder = opt.data_folder
    dataset_name = opt.dataset_name
    off_diagonal_penalty = opt.off_diagonal_penalty
    diagonal_penalty = opt.diagonal_penalty
    share_correlation_penalty = opt.share_correlation_penalty

    if share_correlation_penalty:
        diagonal_penalty = off_diagonal_penalty
        opt.diagonal_penalty = off_diagonal_penalty


    if not os.path.exists(folder):
        print "creating {}...".format(folder)
        os.mkdir(folder)


    n_outputs = 10

    # Natural gradient descent. For now I won't play with these kind of hyper-parameters.
    hyperparameters = dict(
        # "eigh" on covariance matrix or "svd" on data matrix
        decomposition="svd",
        # whether to remix after whitening
        zca=True,
        # compute fisher based on supervised "loss" or model "output"
        objective="output",
        # eigenvalue bias
        eigenvalue_bias=eigenvalue_bias,
        variance_bias=variance_bias,
        batch_normalize=batch_normalize,
        share_parameters=False)

    hyperparameters.update(vars(opt))

    datasets = dataset_loader.get_data(data_folder=data_folder, dataset_name=dataset_name, whiten=whiten_inputs) # TODO add CIFAR10

    features = T.matrix("features")
    targets = T.ivector("targets")

    # fisher index
    fisher_index = T.ivector("fisher_index")

    # compilation helpers
    compile_memo = dict()
    def compile(variables=(), updates=()):
        key = (util.tupelo(variables),
               tuple(OrderedDict(updates).items()))
        try:
            return compile_memo[key]
        except KeyError:
            return compile_memo.setdefault(
                key,
                theano.function(
                    [features, targets],
                    variables,
                    updates=updates,
                    on_unused_input="ignore"))

    # compile theano function and compute on select data
    def compute(variables=(), updates=(), which_set=None, subset=None):
        return (
            compile(variables=variables, updates=updates)(
                **datadict(which_set, subset)))

    def datadict(which_set, subset=None):
        dataset = datasets[which_set]
        return dict(
            (source,
             dataset[source]
             if subset is None
             else dataset[source][subset])
            for source in "features targets".split())


    # Inputs
    # downsample input to keep number of parameters low, (7x7)
    x = features
    x = x.reshape((x.shape[0], 1, 28, 28))

    input_dim = 28 * 28
    x = x.flatten(ndim=2)

    # The model
    layers, nb_param = build_prong(input_dim, n_outputs, layers, hyperparameters)
    parameters_by_layer = [[layer[k] for k in ("Wgb" if batch_normalize and
                                               hyperparameters['bn_scale'] else "Wb")] for layer in layers]

    # Determine which parameters to save
    fisher_param_index = np.sort(np.random.choice(np.arange(nb_param), size=fisher_dimension, replace=False))

    # The updates, checks, and output
    updates, checks, h, off_diag_cost, diag_cost = get_updates(layers, x, hyperparameters)

    if hyperparameters["share_parameters"]:
        # remove repeated parameters
        del parameters_by_layer[2:-1]

    # The loss
    logp = h
    cross_entropy = -logp[T.arange(logp.shape[0]), targets]
    cost = cross_entropy.mean(axis=0)
    all_cost = cost
    all_cost += off_diagonal_penalty * off_diag_cost + diagonal_penalty * diag_cost

    # Get fisher for all the parameters with respect to either the logp or the cross_entropy.
    fisher = get_fisher(parameters_by_layer, n_outputs, logp, cross_entropy, hyperparameters, fisher_index)
    fisher_fn = theano.function([features, fisher_index], fisher)

    #steprule = steprules.rmsprop(scale=lr)
    steprule = steprules.sgd(lr=lr)

    parameters = list(itertools.chain(*parameters_by_layer))
    gradients = OrderedDict(zip(parameters, T.grad(all_cost, parameters)))
    steps = []
    step_updates = []
    for parameter, gradient in gradients.items():
        step, steprule_updates = steprule(parameter, gradient)
        steps.append((parameter, -step))
        step_updates.append((parameter, parameter - step))
        step_updates.extend(steprule_updates)

    # Our metric, fisher matrix (optionial), cross_entropies (train, valid, test), by epoch, and by updates.
    np_fishers = []
    #cross_entropies_by_epoch = {'train': [], 'valid': [], 'test': []}
    #cross_entropies_by_update = {'train': [], 'valid': [], 'test': []}
    #precision_by_epoch = {'train': [], 'valid': [], 'test': []}
    #side_cost_by_epoch = {'train': [], 'valid': [], 'test': []}

    metrics_to_save = {'cross_entropies_by_epoch': {'train': [], 'valid': [], 'test': []},
                       'precision_by_epoch': {'train': [], 'valid': [], 'test': []},
                       'off_diag_cost_by_epoch': {'train': [], 'valid': [], 'test': []},
                       'diag_cost_by_epoch': {'train': [], 'valid': [], 'test': []}}

    for i in xrange(nepochs):

        if save_fisher:
            if shuffle:
                print "shuffling..."
                fisher_param_index = np.sort(np.random.choice(np.arange(nb_param), size= fisher_dimension, replace=False))
            tmp_fisher = None

            for no_batch, a in zip(range(sample_size), range(0, len(datasets["train"]["features"]), batch_size)):

                b = a + batch_size

                # Get the fisher for the index that we follow
                fisher_matrix = fisher_fn(datasets["train"]["features"][a:b],
                                          fisher_param_index.astype('int32'))

                if tmp_fisher is None:
                    tmp_fisher = fisher_matrix
                else:
                    tmp_fisher += fisher_matrix

            np_fishers.append(tmp_fisher / sample_size)

        # Go over the whole dataset once to get the cost.
        for set_name in ['train', 'valid', 'test']:
            
            #cross entropy
            metrics_to_save['cross_entropies_by_epoch'][set_name].append(compute(cost, which_set=set_name))

            # Side cost
            side_cost_1 = compute(off_diag_cost, which_set=set_name)
            side_cost_2 = compute(diag_cost, which_set=set_name)

            metrics_to_save['off_diag_cost_by_epoch'][set_name].append(side_cost_1)
            metrics_to_save['diag_cost_by_epoch'][set_name].append(side_cost_2)

            # Precision
            preds = compute(logp, which_set=set_name).argmax(axis=1)
            #import ipdb; ipdb.set_trace()
            precision = [p == t for p, t in zip(preds,
                                            datasets[set_name]['targets'])]
            precision = np.array(precision).mean()
            metrics_to_save['precision_by_epoch'][set_name].append(precision)

        print i, "train cross entropy", metrics_to_save['cross_entropies_by_epoch']['train'][-1]
        print i, "train precision: {}, valid precision: {}".format(
            metrics_to_save['precision_by_epoch']['train'][-1],
            metrics_to_save['precision_by_epoch']['valid'][-1])

        print i, "training"

        for no_batch, a in enumerate(range(0, len(datasets["train"]["features"]), batch_size)):

            # Do parametrization
            if interval > 0 and no_batch % interval == 0:
                compute(updates=updates, which_set="train", subset=slice(0, estimation_size))

            b = a + batch_size
            compute(updates=step_updates, which_set="train", subset=slice(a, b))

            ## Some random subsampling of the cost
            #for set_name in ['train', 'valid', 'test']:
            #    random_index = np.random.choice(np.arange(len(datasets[set_name])), size=batch_size)
            #    cross_entropies_by_update[set_name].append(compute(cost, which_set=set_name, subset=random_index))

            sys.stdout.write(".")
            sys.stdout.flush()

        print
        print i, "done"

    for set_name in ['train', 'valid', 'test']:
        metrics_to_save['cross_entropies_by_epoch'][set_name].append(compute(cost, which_set=set_name))

    identifier = abs(hash(frozenset(hyperparameters.items())))
    np.savez_compressed(os.path.join(folder, "fishers_{}.npz".format(identifier)),
                        fishers=np.asarray(np_fishers),
                        #cross_entropies_by_epoch=cross_entropies_by_epoch,
                        #precision_by_epoch=precision_by_epoch,
                        **metrics_to_save
                        )

    yaml.dump(hyperparameters,
              open(os.path.join(folder, "hyperparameters_{}.yaml".format(identifier)), "w"))

    print "For {}, we have saved in: {}".format(hyperparameters.items(), identifier)

if __name__ == '__main__':
    main()
