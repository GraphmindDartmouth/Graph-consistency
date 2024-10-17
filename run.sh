datasets=("PROTEINS" "IMDB-BINARY" "IMDB-MULTI" "COLLAB" "NCI1" "NCI109" "COIL-RAG" "DD" "ogbg-molhiv" "reddit_threads")
models=("GCN" "GIN" "GraphSAGE" "GTransformer" "GMT")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo python exp_run.py --model=$model --dataset=$dataset
        # echo python run_dist.py --model=$model --dataset=$dataset
    done
done

# python show_table.py

screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=PROTEINS --device=5"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=IMDB-BINARY --device=5"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=IMDB-MULTI --device=5"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=COLLAB --device=5"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=NCI1 --device=6"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=NCI109 --device=6"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=COIL-RAG --device=6"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=DD --device=6"
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=ogbg-molhiv --device=7" 
screen -dmS GEexp bash -c "python run_dist.py --model=GMT --dataset=reddit_threads --device=7"

screen -dmS GEexp bash -c "python exp_run.py --model=GMT --dataset=reddit_threads --device=0"

screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=PROTEINS --device=3"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=IMDB-BINARY --device=3"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=IMDB-MULTI --device=3"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=COLLAB --device=3"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=NCI1 --device=4"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=NCI109 --device=4"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=COIL-RAG --device=4"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=DD --device=4"
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=ogbg-molhiv --device=2" 
screen -dmS GEexp bash -c "python run_dist.py --model=GTransformer --dataset=reddit_threads --device=2"