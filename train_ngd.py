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
import mnist




def build_prong(input_dim, n_outputs, layers):

    dims = [input_dim] + layers + [n_outputs]

    layers = [
        dict(f=activation.tanh,
             c=util.shared_floatx((m,),   initialization.constant(0)),   # input mean
             U=util.shared_floatx((m, m), initialization.identity()),    # input whitening matrix
             W=util.shared_floatx((m, n), initialization.orthogonal()),  # weight matrix
             g=util.shared_floatx((n,),   initialization.constant(1)),   # gammas (for batch normalization)
             b=util.shared_floatx((n,),   initialization.constant(0)))   # bias
        for m, n in util.safezip(dims[:-1], dims[1:])]
    layers[-1]["f"] = activation.logsoftmax

    return layers

def get_updates(layers, h, hyperparameters):


    # reparametrization updates
    reparameterization_updates = []
    # theano graphs with assertions & breakpoints, to be evaluated after
    # performing the updates
    reparameterization_checks = []

    for i, layer in enumerate(layers):
        f, c, U, W, g, b = [layer[k] for k in "fcUWgb"]

        # construct reparameterization graph
        updates, checks = whitening.get_updates(
            h, c, U, V=W, d=b,
            decomposition="svd", zca=True, bias=hyperparameters['eigenvalue_bias'])
        reparameterization_updates.extend(updates)
        reparameterization_checks.extend(checks)

        # whiten input
        h = T.dot(h - c, U)
        # compute layer as usual
        h = T.dot(h, W)
        if hyperparameters['batch_normalize']:
            h -= h.mean(axis=0, keepdims=True)
            h /= T.sqrt(h.var(axis=0, keepdims=True) + hyperparameters["variance_bias"])
            h *= g
        h += b
        h = f(h)

    return reparameterization_updates, reparameterization_checks, h

def get_fisher(parameters_by_layer, n_outputs, logp, cross_entropy, hyperparameters, sample_size):

    fisher = None

    def estimate_fisher(outputs, n_outputs, parameters):
        # shape (sample_size, n_outputs, #parameters)
        grads = T.stack(*[util.batched_flatcat(
            T.jacobian(outputs[:, j], parameters))
            for j in xrange(n_outputs)])
        # ravel the batch and output axes so that the product will sum
        # over the outputs *and* over the batch. divide by the batch
        # size to get the batch mean.
        grads = grads.reshape((grads.shape[0] * grads.shape[1], grads.shape[2]))
        fisher = T.dot(grads.T, grads) / grads.shape[0]
        return fisher

    # estimate_fisher will perform one backpropagation per sample, so
    # don't go wild

    # don't include visible layer parameters; too many and not full rank
    fisher_parameters = list(itertools.chain(*parameters_by_layer[1:]))
    if hyperparameters["objective"] == "loss":
        # fisher on loss
        fisher = estimate_fisher(cross_entropy[:sample_size, np.newaxis],
                                 1, fisher_parameters)
    elif hyperparameters["objective"] == "output":
        # fisher on output
        fisher = estimate_fisher(logp[:sample_size, :],
                                 n_outputs, fisher_parameters)

    return fisher

def build_parser():
    parser = argparse.ArgumentParser(
        description="Model for Natural Gradient descent")

    # Model specific options
    parser.add_argument('--layers', nargs='+', default=[500, 300, 100], type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--folder', default='./')
    parser.add_argument('--reduction', default=False)
    parser.add_argument('--share', default=False)
    parser.add_argument('--bn', default=False, type=bool)
    parser.add_argument('--fisher', dest='fisher', action='store_true')
    parser.add_argument('--no-fisher', dest='fisher', action='store_false')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--optimizer', default="ngd")
    parser.add_argument('--sample_size', default=100, type=int)
    parser.add_argument('--interval', default=100, type=int)

    return parser

def parse_args(argv):
    opt = build_parser().parse_args(argv)
    return opt

def main(argv=None):

    opt = parse_args(argv)

    layers = opt.layers
    nepochs = opt.epochs
    batch_size = opt.batch_size
    folder = opt.folder
    batch_normalize = opt.bn
    share_parameters = opt.share
    do_reduction = opt.reduction
    save_fisher = opt.fisher
    lr = opt.lr
    sample_size = opt.sample_size
    interval = opt.interval
    optimizer = opt.optimizer

    if share_parameters:
        if len(set(layers)) != 1:
            print "With the share parameters options, all the layers need to have the same size."
            sys.exit(2)

    if not os.path.exists(folder):
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
        eigenvalue_bias=1e-3,
        variance_bias=1e-8,
        batch_normalize=batch_normalize,
        whiten_inputs=True,
        share_parameters=share_parameters)

    datasets = mnist.get_data()

    features = T.matrix("features")
    targets = T.ivector("targets")

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

    if do_reduction:
        reduction = 4
        x = T.nnet.conv.conv2d(
            x,
            (np.ones((1, 1, reduction, reduction),
                     dtype=np.float32)
             / reduction**2),
            subsample=(reduction, reduction))
        input_dim = (28 / reduction)**2

    x = x.flatten(ndim=2)

    # The model
    layers = build_prong(input_dim, n_outputs, layers)
    parameters_by_layer = [[layer[k] for k in ("Wgb" if batch_normalize else "Wb")] for layer in layers]

    # The updates, checks, and output
    updates, checks, h = get_updates(layers, x, hyperparameters)

    if hyperparameters["share_parameters"]:
        # remove repeated parameters
        del parameters_by_layer[2:-1]

    # The loss
    logp = h
    cross_entropy = -logp[T.arange(logp.shape[0]), targets]
    cost = cross_entropy.mean(axis=0)

    # Get fisher for all the parameters with respect to either the logp or the cross_entropy.
    fisher = get_fisher(parameters_by_layer, n_outputs, logp, cross_entropy, hyperparameters, sample_size)

    #steprule = steprules.rmsprop(scale=lr)
    steprule = steprules.sgd(lr=lr)

    parameters = list(itertools.chain(*parameters_by_layer))
    gradients = OrderedDict(zip(parameters, T.grad(cost, parameters)))
    steps = []
    step_updates = []
    for parameter, gradient in gradients.items():
        step, steprule_updates = steprule(parameter, gradient)
        steps.append((parameter, -step))
        step_updates.append((parameter, parameter - step))
        step_updates.extend(steprule_updates)

    np_fishers = []
    cross_entropies = []
    for i in xrange(nepochs):

        if save_fisher:
            # Need to get a subset, othersize, we have memory issue.
            np_fishers.append(compute(fisher, which_set="train", subset=slice(0, sample_size)))

        cross_entropies.append(compute(cost, which_set="train"))
        print i, "train cross entropy", cross_entropies[-1]
        print i, "training"

        for no_batch, a in enumerate(range(0, len(datasets["train"]["features"]), batch_size)):

            # Do parametrization
            if interval > 0 and no_batch % interval == 0:
                compute(updates=updates, which_set="train", subset=slice(0, 1000))

            b = a + batch_size
            compute(updates=step_updates, which_set="train", subset=slice(a, b))
            sys.stdout.write(".")
            sys.stdout.flush()

        print
        print i, "done"

    cross_entropies.append(compute(cost, which_set="train"))

    identifier = abs(hash(frozenset(hyperparameters.items())))
    np.savez_compressed(os.path.join(folder, "fishers_{}.npz".format(identifier)),
                        fishers=np.asarray(np_fishers),
                        cross_entropies=np.asarray(cross_entropies))

    yaml.dump(hyperparameters,
              open(os.path.join(folder, "hyperparameters_{}.yaml".format(identifier)), "w"))

if __name__ == '__main__':
    main()
