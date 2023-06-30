import experiments.train_eco as train
import experiments.test_eco as test

train.run_with_vars(20, 'min_cover', 'ER', 'eco')
test.run_with_params(20, 'min_cover', 'ER', 'eco')

train.run_with_vars(40, 'min_cover', 'ER', 'eco')
test.run_with_params(40, 'min_cover', 'ER', 'eco')

train.run_with_vars(60, 'min_cover', 'ER', 'eco')
test.run_with_params(60, 'min_cover', 'ER', 'eco')

