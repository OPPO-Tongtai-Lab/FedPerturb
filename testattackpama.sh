# for attrate in 0.001 0.005 0.01 0.05 0.1 0.2 0.5; do
# for attrate in 0.02; do
#     # for attack in minmax  minsum  fedperturb alie ipm  labelflipping; do
#     # for attack in alie  ipm  labelflipping; do
#     for attscal in 1.1 1.3 1.7 1.9 2.5 3 5; do
#     # for attscal in 2.5 3 5; do
#         for agg in mean  signguard  dnc centeredclipping fltrust flame; do
#         # for agg in fltrust; do
#             python main.py --agg "$agg" --attack fedperturb --num_byzantine 10 --lr 0.1 --attrate "$attrate" --attscal "$attscal" --model resnet18 --global_round 100 --optimizer "SGD" --dataset cifar10
#         done
#     done
# done

for lr in 0.001 0.005  0.01 0.05  ; do
    # for attack in minmax  minsum  fedperturb alie ipm  labelflipping; do
    # for attack in alie  ipm  labelflipping; do
    for attack in alie ; do
        for agg in mean; do
        # for agg in fltrust; do
            python main.py --agg "$agg" --attack "$attack" --num_byzantine 0 --lr "$lr" --attrate 0.02 --model resnet18 --global_round 160 --local_round 1 --optimizer "SGD" --dataset cifar10
        done
    done
done