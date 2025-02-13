import argparse

if __name__ == '__main__':
    # Set device to CPU since CUDA is not available
    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", default=1024, type=int)
    parser.add_argument("--X_file", default='./data/X_file.txt')
    parser.add_argument("--Y_file", default='./data/Y_file.txt')

    parser.add_argument("--learning_rate", default=0.5, type=float)
    parser.add_argument("--batch_size", default=1024)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--model_weights", default='./weights/model_weights.pth')

    args = parser.parse_args()

    # First generate the training data
    import step.data_gen
    step.data_gen.run(args)

    # Then train the model
    import step.train_nn
    step.train_nn.run(args)

    # Finally deploy/test
    import step.deploy
    step.deploy.run(args)





