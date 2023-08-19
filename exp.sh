python train.py --device 'cuda:1' --scene "drums" 
python visualize.py --device 'cuda:0' --scene "lego" --visualize_depth --visualize_normals --factor 8
python visualize.py --device 'cuda:4' --scene "drums" --factor 2 --visualize_depth --visualize_normals
python extract_mesh.py --device 'cuda:3' --scene "lego" --factor 2 

python train.py --dataset_name "nerf360" --scene "bicycle" --device 'cuda:0' --factor 4  --white_bkgd
python visualize.py --device 'cuda:3' --dataset_name "nerf360" --scene "bicycle" --factor 8 --visualize_depth --visualize_normals --white_bkgd
python train.py --dataset_name "nerf360" --scene "bicycle" --device 'cuda:0' --factor 8  --white_bkgd --min_deg -8 --max_deg 8

python train.py --device 'cuda:0' --scene "drums"  --min_deg -1 --max_deg 15
python train.py --device 'cuda:1' --scene "drums"

python train.py --device 'cuda:7' --scene "drums" --limit_f
python train.py --device 'cuda:8' --scene "drums" 
python train.py --device 'cuda:9' --scene "drums" --use_exp

python train.py --device 'cuda:7' --scene "drums" --sample 'prob'
python train.py --device 'cuda:6' --scene "drums" --learnable_f --sample 'prob'


python train.py --device 'cuda:9' --scene "drums" --learnable_f
