cmd="$1"
echo $cmd


pip3 install -e ".[train]"
pip3 install ninja
pip3 install flash-attn --no-build-isolation

$cmd

