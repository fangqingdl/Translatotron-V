export prefix="/data/workspace/Translatotron-V"

vq_codebook_size=8192
vq_codebook_dim=64
dim=256
num_layers=4
batch_size=4
grad_accum_every=16
save_name="en"
image_dir="data-build/iwslt14.de-en-images/train_en"

torchrun --nproc_per_node=2 --master_port=29675 $prefix/src/train_mgpu.py \
    --output_dir $prefix/image-tokenizer/$save_name \
    --vq_codebook_size $vq_codebook_size \
    --vq_codebook_dim $vq_codebook_dim \
    --data_dir $prefix/$image_dir \
    --patch_size 16 \
    --dim $dim \
    --num_layers $num_layers \
    --batch_size $batch_size \
    --grad_accum_every $grad_accum_every \
    --vae_weight $prefix/image-tokenizer/en/vae.11000.pt
