python3 generate_config.py > generate_config.log
for i in {1..10}; do
    echo "$i"
    cd ./$i
    metad_gentraj
    metad_target_distr
    cd ../
done
