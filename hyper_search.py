import train_ngd
import argparse
import json
import numpy as np

def build_parser():

    parser = train_ngd.build_parser()

    # Model specific options
    parser.add_argument('--explore', default='""', type=str, help="The variable that we want to explore (sample uniformily between the bound). "
                                                                "It needs to have the form [('param', fix_value), ('param2', (lower_bound, upper_bound))]. "
                                                                "Will overrride train_ngd.py parameters. For the float parameters, we do a search in the log space.")
    parser.add_argument('--nb-experiments', type=int, default=1, help="The number of experiment to launch")

    return parser

def parse_args(argv):
    opt = build_parser().parse_args(argv)
    return opt


def main(argv=None):

    # Get the bound
    opt = parse_args(argv)
    explore = json.loads(opt.explore)
    nb_experiments = opt.nb_experiments

    setting = vars(opt)

    # The subscript don't need to know these (and it fucks with the hash)
    del setting['explore']
    del setting['nb_experiments']

    for no_exp in range(nb_experiments):

        # Going over the variables and select there value
        for variable, bound in explore:

            value = bound


            if type(bound) == list and len(bound) == 2:
                min_value, max_value = bound

                if min_value > max_value:
                    raise ValueError("The minimum value is bigger than the maxium value for {}, {}".format(value, bound))

                # sampling the value
                if type(min_value) == int and type(max_value) == int:
                    value = min_value + np.random.random_sample() * (max_value - min_value)
                    value = int(np.round(value))
                else:
                    value = np.exp(np.log(min_value) + (np.log(max_value) - np.log(min_value)) * np.random.random_sample())
                    value = float(value)

            if variable not in setting:
                raise ValueError("The parameter {} is nor defined.".format(variable))

            setting[variable] = value

        #launch an experiment:
        print "Will launch the experiment with the following hyper-parameters: {}".format(setting)
        train_ngd.main(opt)

if __name__ == "__main__":
    main()
