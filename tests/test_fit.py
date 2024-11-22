from wgrp.model import wgrp_model

random_values = [10, 12, 15, 4, 56, 12, 22, 13]


def test_fit():
    model = wgrp_model()
    model.fit(random_values)

    assert model


def test_predict():
    model = wgrp_model()
    model.fit(random_values)
    pred = model.predict(3)

    assert len(pred) > 0
