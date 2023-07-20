python train.py --device 'cuda:1' --scene "drums" 
python visualize.py --device 'cuda:0' --scene "lego" --visualize_depth --visualize_normals --factor 8
python visualize.py --device 'cuda:4' --scene "drums" --factor 2 --visualize_depth --visualize_normals
python extract_mesh.py --device 'cuda:3' --scene "lego" --factor 2 

python train.py --dataset_name "nerf360" --scene "bicycle" --device 'cuda:0' --factor 4  --white_bkgd
python visualize.py --device 'cuda:3' --dataset_name "nerf360" --scene "bicycle" --factor 8 --visualize_depth --visualize_normals --white_bkgd