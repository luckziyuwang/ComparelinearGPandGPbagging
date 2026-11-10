from data import make_dataset

data_a = make_dataset(
    n=20,
    m=500,
    test_size=2000,
    sigma=0.5,
    seed=42,
)

data_b = make_dataset(
    n=20,
    m=500,
    test_size=2000,
    sigma=2.0,
    seed=42,
)

# Same underlying target and inputs.
assert (data_a.w_opt == data_b.w_opt).all()
assert (data_a.X_train == data_b.X_train).all()
assert (data_a.X_test == data_b.X_test).all()
assert (data_a.y_test == data_b.y_test).all()

# Different noise levels should produce different training labels.
assert not (data_a.y_train == data_b.y_train).all()

# Check the expected dimensions.
assert data_a.X_train.shape == (500, 20)
assert data_a.y_train.shape == (500,)
assert data_a.X_test.shape == (2000, 20)
assert data_a.y_test.shape == (2000,)

print("All data generator tests passed!")