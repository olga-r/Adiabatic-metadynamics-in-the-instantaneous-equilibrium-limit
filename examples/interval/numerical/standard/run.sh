
for dir in */; do
  echo  $dir
  cd $dir
  for ddir in */; do
    echo  $ddir
    cd $ddir
    ./run.sh
    cd ../
  done
  cd ../
done
