from data import generate_all
from cvae import train_cvae
from cgan import train_cgan


def main():
    # generate_all()
    train_cvae()
    train_cgan()
