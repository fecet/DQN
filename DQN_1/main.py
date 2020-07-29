if __name__ == '__main__':
    from train import train, model
    import sys
    data_dir = sys.argv[1]
    epochs = int(sys.argv[2])
    train(model, epochs, data_dir)







