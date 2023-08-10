for num_byzantine in 10; do
    # for attack in minmax  minsum  fedperturb alie ipm  labelflipping; do
    # for attack in alie  ipm  labelflipping; do
    for attack in fedperturb ; do
        for agg in mean  flame signguard  dnc centeredclipping fltrust  median  multikrum; do
        # for agg in fltrust; do
            python main.py --agg "$agg" --attack "$attack" --num_byzantine "$num_byzantine" --lr 0.01 --attrate 1.0 --attscal 1.3 --model cnn --global_round 60 --optimizer "Adam" --dataset famnist --local_round 2
        done
        done
    done
done

# for lr in 0.00005 0.0001 0.0005  0.001 0.005  ; do
#     # for attack in minmax  minsum  fedperturb alie ipm  labelflipping; do
#     # for attack in alie  ipm  labelflipping; do
#     for attack in alie ; do
#         for agg in median; do
#         # for agg in fltrust; do
#             python main.py --agg "$agg" --attack "$attack" --num_byzantine 0 --lr "$lr" --attrate 0.02 --model cnn_bn --global_round 100  --optimizer "Adam" --dataset famnist --local_round 1
#         done
#     done
# done
