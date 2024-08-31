jupytext_sync () {
    if [[ $1 == *.py ]]; then
        jupytext --to ipynb $1
        jupytext --set-formats ipynb,py $1
        jupytext --sync $1
    else
        echo "Please provide a .py file"
    fi
}

jupytext_sync $1
