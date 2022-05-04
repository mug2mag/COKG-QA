python KG_dataLoader.py
python schema_dataloader.py

python /benchmarks/gethered/n-n.py
python /benchmarks/gethered_type/n-n.py

CUDA_VISIBLE_DEVICES=0 python train_complex_mydata.py
CUDA_VISIBLE_DEVICES=0 python schema_train_complex_mydata.py

