# import experiments.ER_20spin.test.test_eco_mvc as test20
# import experiments.ER_40spin.test.test_eco_mvc as test40
import experiments.ER_60spin.test.test_eco_mvc as test60
import experiments.ER_100spin.test.test_eco_mvc as test100
import experiments.ER_200spin.test.test_eco_mvc as test200

# import experiments.ER_20spin.train.train_eco_mvc as train20
# import experiments.ER_40spin.train.train_eco_mvc as train40
import experiments.ER_60spin.train.train_eco_mvc as train60
import experiments.ER_100spin.train.train_eco_mvc as train100
import experiments.ER_200spin.train.train_eco_mvc as train200

# train20.run()
# test20.run()

# train40.run()
# test40.run()

train60.run()
test60.run()

train100.run()
test100.run()

train200.run()
test200.run()

