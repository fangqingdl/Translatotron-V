
export prefix="/data/workspace/Translatotron-V"


export save_name=data-build/iwslt14.de-en-lmdb

python $prefix/data-build/create_lmdb_mulproc.py \
    --output_dir $prefix/$save_name \
    --text_data_dir $prefix/data-build/iwslt14.de-en \
    --image_data_dir $prefix/data-build/iwslt14.de-en-images \
    --src_lang de \
    --tgt_lang en \
    --num_workers 64
