import logging
import matplotlib.pyplot as plt
import seaborn as sns

# consider init function to take args
default_level = logging.INFO
logging.basicConfig(format='%(asctime)s::%(name)s::%(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def new_logger(name, level=default_level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def plot_bounds(anchor):
    ax = sns.lineplot(x=list(range(len(anchor.lbs))), y=anchor.lbs, color="black")
    sns.lineplot(x=list(range(len(anchor.ubs))), y=anchor.ubs, color="orange")
    f = plt.gcf()
    f.savefig(f"anchor.png")
    plt.clf()