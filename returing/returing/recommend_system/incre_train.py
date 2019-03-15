from returing.recommend_system.train_model import build_model, data_generator

def incre_train_model():
    model_weights_path = 'w2v.model'
    model = build_model()
    model.summary()

    model.load_weights(model_weights_path)

    V = 10
    window_size = 3

    epochs = 100
    workers = 4
    num_samples = 50
    batch_size = 10
    steps_per_epoch = int(num_samples / batch_size)

    model.fit_generator(data_generator(V=V,
                                       window_size=window_size,
                                       batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        workers=workers,
                        use_multiprocessing=True,
                        verbose=1
                        )

    model.save_weights("w2v.model.new")


if __name__ == '__main__':
    incre_train_model()