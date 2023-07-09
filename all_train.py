import experiments.train_eco as train
import experiments.test_eco as test

train.run_with_vars(20, 'min_cover', 'ER', 'eco')
test.run_with_params(20, 'min_cover', 'ER', 'eco')

train.run_with_vars(100, 'min_cover', 'ER', 'eco')
test.run_with_params(100, 'min_cover', 'ER', 'eco')

