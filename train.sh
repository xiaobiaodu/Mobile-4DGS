CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/bicycle python train.py -s ../datasets/mipnerf360/bicycle -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --grad_abs_thresh 0.0012
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/flowers python train.py -s ../datasets/mipnerf360/flowers -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --dense 0.005 --grad_abs_thresh 0.0015
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/garden python train.py -s ../datasets/mipnerf360/garden -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008    --num_mc_points  1024
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/stump python train.py -s ../datasets/mipnerf360/stump -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --dense 0.004 --grad_abs_thresh 0.0015
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/treehill python train.py -s ../datasets/mipnerf360/treehill -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --dense 0.01 --grad_abs_thresh 0.002
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/room python train.py -s ../datasets/mipnerf360/room -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0008
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/counter python train.py -s ../datasets/mipnerf360/counter -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0008 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/kitchen python train.py -s ../datasets/mipnerf360/kitchen -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0006    --num_mc_points  1024
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=mipnerf360/bonsai python train.py -s ../datasets/mipnerf360/bonsai -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0006  --num_mc_points  1024
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=tt/truck python train.py -s ../datasets/tandt/truck --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=tt/train python train.py -s ../datasets/tandt/train --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=dp/playroom python train.py -s ../datasets/db/playroom --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=dp/drjohnson python train.py -s ../datasets/db/drjohnson --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7 


CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/bicycle --skip_train   --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/flowers --skip_train   --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/garden --skip_train    --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/stump --skip_train      --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/treehill --skip_train   --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/room --skip_train       --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/counter --skip_train     --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/kitchen --skip_train     --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/mipnerf360/bonsai --skip_train       --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/tt/truck --skip_train --mult 0.7     --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/tt/train --skip_train --mult 0.7      --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/dp/playroom --skip_train --mult 0.7     --decode
CUDA_VISIBLE_DEVICES=0 python render.py -m output/dp/drjohnson --skip_train --mult 0.7     --decode

CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/bicycle
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/flowers
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/garden
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/stump
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/treehill
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/room
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/counter
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/kitchen
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/mipnerf360/bonsai
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/tt/truck
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/tt/train
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/dp/playroom
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/dp/drjohnson




bash train_dynamic.sh coffee_martini
bash train_dynamic.sh cook_spinach
bash train_dynamic.sh cut_roasted_beef
bash train_dynamic.sh flame_salmon_1
bash train_dynamic.sh flame_steak
bash train_dynamic.sh sear_steak

