datasets=("PROTEINS" "IMDB-BINARY" "IMDB-MULTI" "COLLAB" "NCI1" "NCI109" "COIL-RAG" "DD" "ogbg-molhiv" "reddit_threads")
models=("GCN" "GIN" "GraphSAGE" "GTransformer" "GMT")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        python exp_run.py --model=$model --dataset=$dataset
        python run_dist.py --model=$model --dataset=$dataset
    done
done

python show_table.py