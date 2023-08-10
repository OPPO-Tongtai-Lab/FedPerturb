#!/opt/conda/bin/python bash

# test for iid 
for num_byzantine in 10 5 2 1 0; do
    for attack in fedperturb; do
    # for attack in minmax  minsum   alie ipm  labelflipping; do
        for agg in mean  signguard  dnc centeredclipping fltrust flame median multikrum  ; do # 
            python main.py --agg "$agg" --attack "$attack" --num_byzantine "$num_byzantine" --lr 0.1 --attrate 0.02 --attscal 1.5 --model resnet18 --global_round 100 --optimizer "SGD" --dataset cifar10 
        done
    done
done


# test for non-iid
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    for num_byzantine in 10 5 2 1 0; do
        for attack in fedperturb; do
        # for attack in minmax  minsum   alie ipm  labelflipping; do
            for agg in mean  signguard  dnc centeredclipping fltrust flame median multikrum  ; do # 
                python main.py --agg "$agg" --attack "$attack" --num_byzantine "$num_byzantine" --lr 0.1 --attrate 0.02 --attscal 1.5 --model resnet18 --global_round 100 --optimizer "SGD" --dataset cifar10 --non_iid --alpha "$alpha"
            done
        done
    done
done

# test for lr
for lr in 0.05 0.01 0.005 0.001 0.0005; do
    for attack in fedperturb; do
    # for attack in minmax  minsum   alie ipm  labelflipping; do
        for agg in mean  signguard  dnc centeredclipping fltrust flame; do
            python main.py --agg "$agg" --attack "$attack" --num_byzantine 0 --lr "$lr" --attrate 0.02 --model resnet18 --global_round 200 --optimizer "SGD" --dataset cifar10
        done
    done
done


